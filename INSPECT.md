# 🔍 SemBench Inspection Notes

> **Session summary**: Deep-dive into SemBench benchmark structure, input/output formats, and architectural options for running semantic SQL queries on a lakehouse (Presto/Spark).

---

## Table of Contents

1. [What is SemBench?](#1-what-is-sembench)
2. [Concrete Example: Input → Process → Output](#2-concrete-example-input--process--output)
3. [Input Format Clarification](#3-input-format-clarification)
4. [Loading Data into a Lakehouse](#4-loading-data-into-a-lakehouse)
5. [How Audio and Image Files are Stored](#5-how-audio-and-image-files-are-stored)
6. [Two Approaches to Semantic SQL on a Lakehouse](#6-two-approaches-to-semantic-sql-on-a-lakehouse)
7. [Data Movement Problem for Large Datasets](#7-data-movement-problem-for-large-datasets)
8. [UDFs in Presto: How Common?](#8-udfs-in-presto-how-common)
9. [Streaming / Micro-batch Processing with Presto](#9-streaming--micro-batch-processing-with-presto)
10. [pandas_udf Special Features](#10-pandas_udf-special-features)
11. [The Two-Layer Execution Architecture](#11-the-two-layer-execution-architecture)
12. [Operator Classification: Layer 1 vs Layer 2](#12-operator-classification-layer-1-vs-layer-2)

---

## 1. What is SemBench?

SemBench is a benchmark targeting **semantic query processing engines** — systems that extend SQL with LLM-powered operators over multi-modal data (tables, text, images, audio).

### Scenarios Overview

| Scenario | #Queries | Modalities | Key Operators |
|---|---|---|---|
| Movie | 10 | Table, Text | Filter, Join, Rank, Classify |
| Wildlife | 10 | Table, Image, Audio | Filter |
| E-Commerce | 14 | Table, Text, Image | Filter, Join, Map, Rank, Classify |
| MMQA | 11 | Table, Text, Image | Filter, Join, Map |
| Cars | 10 | Table, Text, Image, Audio | Filter, Classify |

### Systems Evaluated

- **LOTUS** — semantic operators with cost-guaranteed accuracy
- **Palimpzest** — cost-based optimization
- **ThalamusDB** — approximate query processing
- **BigQuery** — Google's industrial system with native `AI.IF()`

---

## 2. Concrete Example: Input → Process → Output

### Example A — Text Modality (Medical Q1)

#### 📥 Input Data (CSV files)

**`patient_data.csv`** — structured/relational table:
```csv
age,gender,smoking_history,did_family_have_cancer,patient_id
78,Female,Current,0,8079
34,Male,Former,0,2947
```

**`text_symptoms_data.csv`** — unstructured free-text column:
```csv
symptoms,patient_id,symptom_id
"Strong itchiness, chills, nausea, and a high temperature...",9868,853
"I have nasal congestion and a blocked nose...",779,993
```

#### 📝 Input Query (Semantic SQL)

```sql
-- ThalamusDB dialect
SELECT patients.patient_id
FROM patients, symptoms_texts
WHERE patients.patient_id = symptoms_texts.patient_id
  AND NLfilter(symptoms_texts.symptoms, 'Patient has an allergy.')
```

> `NLfilter(column, 'instruction')` sends each row's text to an LLM and returns TRUE/FALSE.

#### 📤 Output

**Ground truth** (`ground_truth/Q1.csv`) — ~50 correct patient IDs:
```csv
age,gender,smoking_history,did_family_have_cancer,patient_id
78,Male,Current,1,4716
69,Male,Former,1,3638
```

**System result** (`bigquery_flash/Q1.csv`) — BigQuery+Gemini Flash returned ~325 IDs (many false positives).

**Evaluation**: Precision / Recall / F1 + latency + API cost ($) + token consumption.

---

### Example B — Image Modality (Medical Q8)

```sql
SELECT patients.patient_id
FROM patients, skin_images
WHERE patients.patient_id = skin_images.patient_id
  AND patients.did_family_have_cancer = 1
  AND NLfilter(skin_images.image_path,
      'This image shows a malignant human skin mole (cancerous/sick).')
LIMIT 100;
```

`NLfilter` here passes the **image file** to an LLM vision model for classification.

---

### Example C — Audio Modality (Cars Q2)

```sql
-- BigQuery dialect
SELECT DISTINCT c.car_id
FROM cars_dataset.cars AS c
JOIN cars_dataset.audio_mm AS a ON c.car_id = a.car_id
WHERE c.fuel_type = 'Electric'
  AND AI.IF(
    prompt => ("Return true if the car has a dead battery.", a.audio_file),
    connection_id => '<<connection>>'
  )
```

---

### 📊 Input/Output Summary Table

| Component | Format | Description |
|---|---|---|
| Input data | CSV files | Structured columns + paths to text/image/audio files |
| Input query | `.sql` (semantic SQL) | SQL extended with AI function calls |
| Semantic operator | `NLfilter(col, 'instruction')` / `AI.IF(prompt)` | LLM evaluates each row |
| Modalities | Table, Text, Image (`.jpg`), Audio (`.wav`) | Passed to LLM as text or vision input |
| System output | CSV | Set of rows (usually IDs) matching the query |
| Ground truth | CSV | Pre-computed correct answer set |
| Evaluation metrics | Precision / Recall / F1, latency, cost ($), tokens | System CSV vs. ground truth CSV |

---

## 3. Input Format Clarification

> ❓ *"Does the input include a natural language question?"*

**No.** The natural language `.txt` files (e.g., `Q1.txt`: *"Find cars that were in a crash."*) are **human-readable documentation only** — they describe what the query does in plain English. They are **not** fed into the system at runtime.

The actual semantic intent is embedded directly inside the SQL query as a string argument to the AI function:

```sql
-- The NL description is embedded IN the SQL, not passed separately
WHERE AI.IF(
    "You are given a complaint. Return true if the car was in a crash.", 
    c.summary
)
```

### Runtime Input = Semantic SQL + Data Files

| | Format | Role |
|---|---|---|
| Semantic SQL | `.sql` file | ✅ **Actual input** to the system |
| CSV tables | `.csv` files | ✅ **Actual input** data |
| Images / Audio | `.jpg`, `.wav` files | ✅ **Actual input** data (referenced by CSV paths) |
| Natural language `.txt` | `.txt` file | 📖 Human-readable annotation only, **not** system input |

---

## 4. Loading Data into a Lakehouse

### Data Loading — Feasible ✅

```sql
-- Spark/Presto: load structured tables
CREATE TABLE patients USING CSV LOCATION 's3://bucket/patient_data.csv';
CREATE TABLE symptoms_texts USING CSV LOCATION 's3://bucket/text_symptoms_data.csv';
CREATE TABLE skin_images USING CSV LOCATION 's3://bucket/image_skin_data.csv';
-- image_path column contains: files/medical/.../810_train_malignant_831.jpg
```

### Semantic SQL — Requires Extension ⚠️

Presto and Spark do **not** natively understand `NLfilter(...)` or `AI.IF(...)`. Three options:

| Option | Description | Feasibility |
|---|---|---|
| **UDF** | Register Python/Java UDF that calls LLM API | ✅ Works, but has limitations |
| **SQL split + Python** | LLM decomposes query into standard SQL + Python DataFrame processing | ✅ Preferred for most cases |
| **Pre-compute labels** | Batch LLM inference first, write labels back, then plain SQL | ⚠️ Breaks for complex operators |

---

## 5. How Audio and Image Files are Stored

### The Pattern: Binary Files → Object Storage, Metadata → Table

From the actual BigQuery setup code (`src/scenario/medical/setup/bigquery.py`):

```python
# Step 1: Upload .wav/.jpg files to GCS bucket
upload_to_gcs(local_file_path, "gs://bucket/lung_audios/BP57_COPD.wav")

# Step 2: Create EXTERNAL TABLE in BigQuery pointing to GCS
CREATE OR REPLACE EXTERNAL TABLE medical_dataset.audios
WITH CONNECTION `us.connection`
OPTIONS (object_metadata = 'SIMPLE', uris = ['gs://bucket/lung_audios/*'])

# Step 3: Join metadata + object reference into one multimodal table
CREATE OR REPLACE TABLE medical_dataset.audio_mm AS
SELECT lung_audio.*, ot.ref AS image   -- 'image' = binary object reference
FROM medical_dataset.lung_audio
INNER JOIN medical_dataset.audios ot ON ot.uri = lung_audio.path
```

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Object Storage (S3 / GCS / ADLS)                          │
│  Actual binary files: .jpg, .wav, .mp3                      │
│  s3://bucket/lung_audios/BP57_COPD.wav                      │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │ URI reference
┌─────────────────────────────────────────────────────────────┐
│  Lakehouse Table (Parquet / Delta / Iceberg)                │
│  patient_id | location | filtration_type | path (URI)       │
│  11056      | Post Right | bell          | s3://bucket/...  │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │ SQL query
┌─────────────────────────────────────────────────────────────┐
│  Query Engine (Presto / Spark / BigQuery)                   │
│  Reads metadata table; passes URI to LLM via UDF            │
└─────────────────────────────────────────────────────────────┘
```

> **Key constraint**: The LLM must be able to access the object storage. BigQuery+Gemini handles this natively. For Spark+OpenAI, the UDF must download bytes and send as base64, or use a model that accepts S3/GCS URIs.

---

## 6. Two Approaches to Semantic SQL on a Lakehouse

### Approach 1 — UDF Inside the Lakehouse Engine

```sql
-- Spark SQL with registered UDF
SELECT patient_id
FROM patients JOIN symptoms_texts USING (patient_id)
WHERE nlfilter_udf(symptoms, 'Patient has an allergy.')
```

### Approach 2 — LLM Splits Query: Standard SQL + Python DataFrame

```python
# Part 1: standard SQL → DataFrame (Spark/Presto executes this)
df = spark.sql("""
    SELECT s.patient_id, s.symptoms
    FROM patients p JOIN symptoms_texts s USING (patient_id)
""")

# Part 2: Python + LLM on the DataFrame (LOTUS/Palimpzest style)
result = df.sem_filter("Patient has an allergy.")
```

### Comparison

| Dimension | Approach 1 (UDF) | Approach 2 (SQL split + Python) |
|---|---|---|
| **Query optimization** | ❌ UDF is black box to planner | ✅ Full control over predicate ordering |
| **Predicate ordering** | ❌ May call LLM before cheap SQL filters | ✅ Cheap SQL always runs first |
| **Batching LLM calls** | ❌ Row-by-row by default | ✅ Natural — full DataFrame in memory |
| **Multi-modal data** | ⚠️ Downloads binary per row during execution | ✅ Parallel download before LLM call |
| **Complex operators (NLjoin)** | ❌ Requires full cross-join first | ✅ Python can implement blocking/sampling |
| **Portability** | ❌ Different UDF API per engine | ✅ Python is engine-agnostic |
| **Debugging** | ❌ Hard inside engine | ✅ Standard Python debugging |
| **Scalability** | ✅ Engine handles distributed execution | ⚠️ DataFrame must fit in driver memory |

> **Recommendation**: Approach 2 is generally preferable. The bottleneck is always the LLM, not the SQL engine. Approach 2 gives full control over batching, ordering, and sampling.

---

## 7. Data Movement Problem for Large Datasets

### The Problem

```python
# This materializes ALL rows into driver memory — dangerous at scale
df = spark.sql("SELECT patient_id, symptoms FROM symptoms_texts").toPandas()
```

| Problem | Impact |
|---|---|
| Driver memory | `toPandas()` collects all rows to one machine |
| Network transfer | Moving from distributed storage to driver is slow |
| LLM throughput | Rate limits become the wall even with batching |
| Multi-modal files | Downloading all binaries amplifies the problem |

### Solutions

#### Solution A — Push-down cheap SQL predicates first
```python
df = spark.sql("""
    SELECT s.patient_id, s.symptoms
    FROM patients p JOIN symptoms_texts s USING (patient_id)
    WHERE p.age > 50                    -- eliminates 60% of rows
      AND p.did_family_have_cancer = 1  -- eliminates another 70%
""").toPandas()
# Only ~12% of rows need LLM inference — 8x less data movement
```

#### Solution B — Distributed pandas_udf (hybrid)
```python
@pandas_udf("boolean")
def nlfilter_batched(texts: pd.Series) -> pd.Series:
    return pd.Series(llm_api.batch_classify(texts.tolist(), instruction))

# Data stays distributed — no driver memory issue
df.filter(nlfilter_batched(df.symptoms))
```

#### Solution C — Approximate Query Processing (ThalamusDB approach)
```python
df_sample = spark.sql("SELECT * FROM symptoms_texts TABLESAMPLE (10 PERCENT)")
# Run LLM on sample → estimate selectivity → decide on full scan vs. approximate result
```

#### Solution D — Streaming / Micro-batch
```python
cursor.execute("SELECT patient_id, symptoms FROM symptoms_texts WHERE ...")
while True:
    rows = cursor.fetchmany(500)   # bounded memory
    if not rows: break
    chunk_df = pd.DataFrame(rows)
    results.append(sem_filter(chunk_df, "Patient has an allergy."))
```

---

## 8. UDFs in Presto: How Common?

### Short Answer
Presto UDFs exist but are **significantly less ergonomic** than Spark. Not a natural fit for LLM-calling operators.

### What Presto Supports

| UDF Type | Capability | LLM feasibility |
|---|---|---|
| Built-in / Lambda | Pure SQL transformations | ❌ Cannot call external APIs |
| SQL-defined (Trino) | Simple SQL functions | ❌ Cannot call external APIs |
| Java plugin JAR | Full Java code | ⚠️ Possible but painful |

### Java UDF Problems for LLM Calls

```java
// Must write Java, compile JAR, deploy to EVERY worker node, restart cluster
public class NLFilterFunction extends ScalarFunction {
    @ScalarFunction("nlfilter")
    @SqlType(StandardTypes.BOOLEAN)
    public static boolean nlFilter(@SqlType(StandardTypes.VARCHAR) Slice text, ...) {
        return callLLMApi(text.toStringUtf8(), ...);  // blocking HTTP call in worker thread!
    }
}
```

Issues: blocking I/O in worker threads, no batching, JAR deployment friction, no Python support, Trino vs. PrestoDB incompatibility.

### Engine Comparison

| Engine | UDF Language | LLM UDF Feasibility |
|---|---|---|
| **Presto / Trino** | Java (plugin JAR) | ⚠️ Possible but painful |
| **Spark** | Python (`pandas_udf`) | ✅ Good |
| **BigQuery** | JS / remote function | ✅ Native (`AI.IF` built-in) |
| **Databricks** | Python / SQL | ✅ Native (`ai_query()` built-in) |
| **DuckDB** | Python / C++ | ✅ Good (FlockMTL extension) |

> **Recommendation**: If your lakehouse runs on Presto, use Presto for cheap SQL filtering, then export to Python or Spark for LLM inference. Do not implement LLM-calling UDFs in Presto.

---

## 9. Streaming / Micro-batch Processing with Presto

> The SQL sent to Presto is **pure standard SQL** — no semantic operators. Presto only filters/joins. Semantic processing happens in Python after each chunk.

### Method 1 — OFFSET/LIMIT Pagination
```python
offset = 0
while True:
    cursor.execute(f"""
        SELECT s.patient_id, s.symptoms
        FROM patients p JOIN symptoms_texts s ON p.patient_id = s.patient_id
        WHERE p.age > 50
        ORDER BY s.patient_id
        LIMIT 1000 OFFSET {offset}
    """)
    rows = cursor.fetchall()
    if not rows: break
    chunk_df = pd.DataFrame(rows, columns=['patient_id', 'symptoms'])
    results.append(sem_filter(chunk_df, "Patient has an allergy."))
    offset += 1000
```

### Method 2 — Partition-based Chunking (more efficient)
```python
for start_id in range(min_id, max_id + 1, 1000):
    cursor.execute(f"""
        SELECT s.patient_id, s.symptoms
        FROM patients p JOIN symptoms_texts s ON p.patient_id = s.patient_id
        WHERE s.patient_id >= {start_id} AND s.patient_id < {start_id + 1000}
    """)
    # ... process chunk
```

### Method 3 — `fetchmany()` on Single Query (simplest ✅)
```python
# Execute once — standard SQL, no semantic operators
cursor.execute("""
    SELECT s.patient_id, s.symptoms
    FROM patients p JOIN symptoms_texts s ON p.patient_id = s.patient_id
    WHERE p.age > 50 AND p.did_family_have_cancer = 1
""")

while True:
    rows = cursor.fetchmany(500)   # streams from Presto result buffer
    if not rows: break
    chunk_df = pd.DataFrame(rows, columns=['patient_id', 'symptoms'])
    results.append(sem_filter(chunk_df, "Patient has an allergy."))
```

> `fetchmany()` is the most practical: Presto executes the query once and streams results back. Driver memory stays bounded at `chunk_size` rows at any time.

---

## 10. `pandas_udf` Special Features

### Regular UDF vs. pandas_udf

```
Regular UDF:   [row1] → Python → [row2] → Python → [row3] → Python ...  (N round-trips)
pandas_udf:    [row1, row2, ..., rowN] → Python  (1 round-trip per partition via Arrow)
```

### Key Features

#### 1. Vectorized Execution — Called Per Partition, Not Per Row
```python
@pandas_udf("boolean")
def nlfilter_batch(texts: pd.Series) -> pd.Series:
    results = []
    for i in range(0, len(texts), 100):
        batch = texts.iloc[i:i+100].tolist()
        results.extend(llm_api.classify_batch(batch, instruction))
    return pd.Series(results)
```

#### 2. Apache Arrow Serialization — Near-Zero Overhead
- ~10-100x faster than Pickle (used by regular UDFs)
- Zero-copy for numeric types
- Same format used internally by Spark

#### 3. Data Stays Distributed — No Driver Memory Bottleneck
```
Regular UDF:   data stays in executors ✅ (but row-by-row Python overhead)
pandas_udf:    data stays in executors ✅ (partition sent to co-located Python worker)
toPandas():    ALL data collected to driver ❌ (memory bottleneck)
```

#### 4. Multiple UDF Types
```python
# Scalar: Series → Series
@pandas_udf("boolean")
def nlfilter(texts: pd.Series) -> pd.Series: ...

# Scalar Iterator: load model ONCE per partition, not per row
@pandas_udf("boolean")
def nlfilter_with_setup(texts: Iterator[pd.Series]) -> Iterator[pd.Series]:
    model = load_model_once()   # expensive setup done once
    for batch in texts:
        yield pd.Series(model.predict(batch.tolist()))
```

#### 5. Concrete Speedup Example
- 10,000 rows, LLM API latency 200ms per call
- Regular UDF: 10,000 × 200ms = **33 minutes** (sequential)
- `pandas_udf` batch_size=100 on 10 executors: (10,000/100/10) × 200ms = **2 seconds**

---

## 11. The Two-Layer Execution Architecture

```
Input: Semantic SQL query
         │
         ▼
    ┌─────────────────────────────────────────────────────┐
    │  LLM Query Decomposer                               │
    │  Splits semantic SQL into Layer 1 + Layer 2         │
    └─────────────────────────────────────────────────────┘
         │                          │
         ▼                          ▼
  ┌─────────────────┐      ┌──────────────────────────────┐
  │  LAYER 1        │      │  LAYER 2                     │
  │  Spark          │─────▶│  Python DataFrame            │
  │  Standard SQL   │      │  (LOTUS/Palimpzest style)    │
  │  + pandas_udf   │      │  Complex semantic operators  │
  └─────────────────┘      └──────────────────────────────┘
```

### What Goes in Each Layer

**Layer 1 — Spark (standard SQL + pandas_udf for simple semantic operators)**
- All standard SQL: joins on structured columns, equality/range filters, GROUP BY, ORDER BY
- `sem_filter`, `sem_map`, `ai_classify` → implemented as `pandas_udf`

**Layer 2 — Python DataFrame (LOTUS/Palimpzest style)**
- `sem_join` — pair-wise LLM comparison
- `sem_agg` — hierarchical reduce
- `sem_topk` — pairwise tournament
- `sem_group_by` — center discovery + assignment

### Concrete Example: Q9 (NLjoin)

```python
# LAYER 1: Spark handles standard SQL joins
skin_df = spark.sql("""
    SELECT p.patient_id, s.image_path AS skin_path
    FROM patients p JOIN skin_images s ON p.patient_id = s.patient_id
""").toPandas()   # small after cheap SQL filters

xray_df = spark.sql("""
    SELECT p.patient_id, x.image_path AS xray_path
    FROM patients p JOIN x_ray_images x ON p.patient_id = x.patient_id
""").toPandas()

# LAYER 2: Python handles NLjoin with blocking
merged = skin_df.merge(xray_df, on='patient_id')  # standard pandas join as blocker
results = [
    row['patient_id']
    for _, row in merged.iterrows()
    if llm_compare_images(row['skin_path'], row['xray_path'],
                          "Both images indicate diseases.")
]
```

---

## 12. Operator Classification: Layer 1 vs Layer 2

### Final Classification Table

| Operator | Layer | Reason |
|---|---|---|
| `sem_filter(l: X → Bool)` | ✅ **Layer 1 — pandas_udf** | Per-row independent: `M(t_i, l)` |
| `sem_map(l: X → Y)` | ✅ **Layer 1 — pandas_udf** | Per-row independent: `M(t_i, l)` |
| `ai_classify(l, labels[])` | ✅ **Layer 1 — pandas_udf** | Per-row independent, constrained output |
| `sem_join(t, l: (X,Y) → Bool)` | ❌ **Layer 2 — Python** | Pair-wise: `M((t_i, t_j), l)`, needs blocking |
| `sem_agg(l: T[X] → X)` | ❌ **Layer 2 — Python** | Hierarchical reduce over all rows |
| `sem_topk(l, k: int)` | ❌ **Layer 2 — Python** | Pairwise tournament comparisons |
| `sem_group_by(l: X → Y, C: int)` | ❌ **Layer 2 — Python** | Center discovery needs all rows; assignment follows in same Python context — cannot go back to Layer 1 |

### The Dividing Line

> **Layer 1**: operators where each row is evaluated **independently** with no prior knowledge of other rows.
>
> **Layer 2**: operators that require **seeing multiple rows together** at any point in their execution — even if a later step within that operator is technically per-row.

### `ai_classify` — Special Note

`ai_classify` is a constrained version of `sem_map` — it returns one of a fixed set of labels instead of free-form output. This makes it:
- ✅ Fully vectorizable as `pandas_udf`
- ✅ More reliable (structured output / constrained decoding)
- ✅ Cheaper (multiple-choice prompt vs. open-ended generation)

```python
@pandas_udf("string")
def ai_classify_udf(texts: pd.Series) -> pd.Series:
    return pd.Series(llm_batch_classify(
        texts.tolist(),
        instruction="Classify the sentiment.",
        labels=['positive', 'negative', 'neutral']
    ))
```

---

*Generated from SemBench inspection session — 2026-02-28*