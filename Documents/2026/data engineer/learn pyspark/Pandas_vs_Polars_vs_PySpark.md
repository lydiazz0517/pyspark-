# Pandas vs Polars vs PySpark 完整对比

## 三大数据处理工具概览

| 特性 | Pandas | Polars | PySpark |
|------|--------|--------|---------|
| **发布时间** | 2008 | 2020 | 2014 |
| **底层语言** | C/Python | Rust | Scala/Java |
| **执行模式** | 立即执行 | 延迟执行 | 延迟执行 |
| **处理速度** | 基准 (1x) | 🚀 **5-10x** | 分布式 |
| **内存效率** | 较低 | 🔥 **极高** | 分布式 |
| **数据规模** | < 10GB | < 100GB | TB+ 级别 |
| **并行处理** | 单线程为主 | 多线程 | 分布式集群 |
| **学习曲线** | ⭐ 简单 | ⭐⭐ 中等 | ⭐⭐⭐ 较难 |
| **生态系统** | 🌟 成熟丰富 | 🌱 快速成长 | 🌟 企业级 |
| **最佳场景** | 原型开发 | 单机大数据 | 分布式大数据 |

---

## 为什么选择 Polars？

### Polars 的优势 🚀

1. **极快的速度** - 比 Pandas 快 5-10 倍
2. **内存高效** - 使用 Apache Arrow 格式
3. **并行处理** - 自动利用所有 CPU 核心
4. **延迟执行** - 像 PySpark 一样优化查询计划
5. **表达式 API** - 更现代、更优雅的语法
6. **无 GIL 限制** - Rust 实现，真正的并行

### 何时使用哪个？

```
数据量 < 5GB    → Pandas (最简单)
数据量 5-50GB   → Polars (最快) ⭐ 推荐
数据量 50-100GB → Polars + Streaming
数据量 > 100GB  → PySpark (分布式)
```

---

## 常用操作三方对比

### 1. 安装

**Pandas:**
```bash
pip install pandas
```

**Polars:**
```bash
pip install polars
```

**PySpark:**
```bash
pip install pyspark
```

---

### 2. 创建 DataFrame

**Pandas:**
```python
import pandas as pd

df = pd.DataFrame({
    'id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [70000, 80000, 90000]
})
```

**Polars:**
```python
import polars as pl

df = pl.DataFrame({
    'id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [70000, 80000, 90000]
})
```

**PySpark:**
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

df = spark.createDataFrame([
    (1, 'Alice', 25, 70000),
    (2, 'Bob', 30, 80000),
    (3, 'Charlie', 35, 90000)
], ['id', 'name', 'age', 'salary'])
```

---

### 3. 查看数据

**Pandas:**
```python
df.head()           # 前 5 行
df.head(10)         # 前 10 行
df.info()           # 数据信息
df.describe()       # 统计描述
df.shape            # (行数, 列数)
```

**Polars:**
```python
df.head()           # 前 5 行
df.head(10)         # 前 10 行
df.describe()       # 统计描述
df.shape            # (行数, 列数)
df.glimpse()        # 类似 info() 的概览 ⭐
```

**PySpark:**
```python
df.show()           # 默认 20 行
df.show(10)         # 前 10 行
df.printSchema()    # 数据结构
df.describe().show() # 统计描述
df.count()          # 行数（需要计算）
```

---

### 4. 过滤数据 ⭐⭐⭐

**Pandas:**
```python
# 单条件
df_filtered = df[df['age'] > 25]

# 多条件
df_filtered = df[(df['age'] > 25) & (df['salary'] > 75000)]

# 使用 query
df_filtered = df.query('age > 25 and salary > 75000')
```

**Polars:**
```python
import polars as pl

# 方法 1：filter() - 推荐 ⭐
df_filtered = df.filter(pl.col('age') > 25)

# 方法 2：多条件 - 超级优雅！
df_filtered = df.filter(
    (pl.col('age') > 25) & (pl.col('salary') > 75000)
)

# 方法 3：链式调用
df_filtered = (
    df
    .filter(pl.col('age') > 25)
    .filter(pl.col('salary') > 75000)
)
```

**PySpark:**
```python
from pyspark.sql.functions import col

# 方法 1：filter() + col()
df_filtered = df.filter(col('age') > 25)

# 方法 2：多条件
df_filtered = df.filter(
    (col('age') > 25) & (col('salary') > 75000)
)
```

**语法对比：**
```python
# Pandas:  df[df['age'] > 25]
# Polars:  df.filter(pl.col('age') > 25)  ← 更清晰
# PySpark: df.filter(col('age') > 25)
```

---

### 5. 选择列

**Pandas:**
```python
# 单列（返回 Series）
df['name']

# 多列（返回 DataFrame）
df[['name', 'age']]
```

**Polars:**
```python
# 单列（返回 Series）
df['name']
df.select('name')

# 多列（返回 DataFrame）
df.select(['name', 'age'])
df.select(pl.col('name'), pl.col('age'))

# 正则选择（高级）⭐
df.select(pl.col('^.*e$'))  # 选择以 'e' 结尾的列
```

**PySpark:**
```python
# 单列
df.select('name')

# 多列
df.select('name', 'age')
df.select(col('name'), col('age'))
```

---

### 6. 添加新列

**Pandas:**
```python
# 方法 1：直接赋值
df['age_plus_10'] = df['age'] + 10

# 方法 2：assign()
df = df.assign(age_plus_10=df['age'] + 10)

# 方法 3：apply()
df['age_category'] = df['age'].apply(lambda x: 'Senior' if x > 30 else 'Junior')
```

**Polars:**
```python
# 方法 1：with_columns() - 推荐 ⭐
df = df.with_columns(
    (pl.col('age') + 10).alias('age_plus_10')
)

# 方法 2：添加多列
df = df.with_columns([
    (pl.col('age') + 10).alias('age_plus_10'),
    (pl.col('salary') * 1.1).alias('salary_increased')
])

# 方法 3：条件列
df = df.with_columns(
    pl.when(pl.col('age') > 30)
      .then(pl.lit('Senior'))
      .otherwise(pl.lit('Junior'))
      .alias('age_category')
)
```

**PySpark:**
```python
# 使用 withColumn()
df = df.withColumn('age_plus_10', col('age') + 10)

# 条件列
from pyspark.sql.functions import when, lit

df = df.withColumn('age_category',
    when(col('age') > 30, lit('Senior'))
    .otherwise(lit('Junior'))
)
```

---

### 7. 分组聚合 ⭐⭐⭐

**Pandas:**
```python
# 单个聚合
df.groupby('name')['salary'].mean()

# 多个聚合
df.groupby('name').agg({
    'age': 'mean',
    'salary': ['sum', 'count', 'mean']
})

# 使用 agg 函数
df.groupby('name').agg(
    avg_age=('age', 'mean'),
    total_salary=('salary', 'sum')
)
```

**Polars:**
```python
# 方法 1：简洁优雅 ⭐
df.groupby('name').agg([
    pl.col('age').mean().alias('avg_age'),
    pl.col('salary').sum().alias('total_salary'),
    pl.col('salary').count().alias('count')
])

# 方法 2：多个聚合
df.groupby('name').agg([
    pl.mean('age'),
    pl.sum('salary'),
    pl.count()
])

# 方法 3：条件聚合
df.groupby('name').agg([
    pl.col('salary').filter(pl.col('age') > 30).mean().alias('avg_salary_senior')
])
```

**PySpark:**
```python
from pyspark.sql.functions import mean, sum, count

df.groupBy('name').agg(
    mean('age').alias('avg_age'),
    sum('salary').alias('total_salary'),
    count('*').alias('count')
)
```

**速度对比：**
```
大数据集分组聚合速度：
Polars > PySpark (单机) > Pandas
  1x      0.8x               0.1x
```

---

### 8. 排序

**Pandas:**
```python
# 升序
df.sort_values('age')

# 降序
df.sort_values('age', ascending=False)

# 多列
df.sort_values(['name', 'age'], ascending=[True, False])
```

**Polars:**
```python
# 升序
df.sort('age')

# 降序
df.sort('age', descending=True)

# 多列
df.sort(['name', 'age'], descending=[False, True])

# 使用表达式 ⭐
df.sort(pl.col('age').cast(pl.Int32))
```

**PySpark:**
```python
from pyspark.sql.functions import desc, asc

# 升序
df.orderBy('age')

# 降序
df.orderBy(desc('age'))

# 多列
df.orderBy('name', desc('age'))
```

---

### 9. JOIN 操作

**Pandas:**
```python
# Inner join
pd.merge(df1, df2, on='id')

# Left join
pd.merge(df1, df2, on='id', how='left')

# 多个键
pd.merge(df1, df2, on=['id', 'name'])
```

**Polars:**
```python
# Inner join
df1.join(df2, on='id')

# Left join
df1.join(df2, on='id', how='left')

# 多个键
df1.join(df2, on=['id', 'name'])

# 高级：使用表达式
df1.join(
    df2,
    left_on='user_id',
    right_on='id',
    how='left'
)
```

**PySpark:**
```python
# Inner join
df1.join(df2, 'id')

# Left join
df1.join(df2, 'id', 'left')

# 多个键
df1.join(df2, ['id', 'name'])
```

---

### 10. 处理缺失值

**Pandas:**
```python
# 删除缺失值
df.dropna()
df.dropna(subset=['age'])

# 填充缺失值
df.fillna(0)
df.fillna({'age': 0, 'name': 'Unknown'})
```

**Polars:**
```python
# 删除缺失值
df.drop_nulls()
df.drop_nulls(subset=['age'])

# 填充缺失值
df.fill_null(0)
df.fill_null({'age': 0, 'name': 'Unknown'})

# 高级填充 ⭐
df.with_columns([
    pl.col('age').fill_null(pl.col('age').mean())
])
```

**PySpark:**
```python
# 删除缺失值
df.dropna()
df.dropna(subset=['age'])

# 填充缺失值
df.fillna(0)
df.fillna({'age': 0, 'name': 'Unknown'})
```

---

### 11. 字符串操作

**Pandas:**
```python
# 转大写
df['name'] = df['name'].str.upper()

# 包含判断
df[df['name'].str.contains('Alice')]

# 分割
df['name'].str.split(' ')
```

**Polars:**
```python
# 转大写
df = df.with_columns(
    pl.col('name').str.to_uppercase().alias('name_upper')
)

# 包含判断
df.filter(pl.col('name').str.contains('Alice'))

# 分割
df = df.with_columns(
    pl.col('name').str.split(' ').alias('name_parts')
)
```

**PySpark:**
```python
from pyspark.sql.functions import upper, split

# 转大写
df = df.withColumn('name_upper', upper(col('name')))

# 包含判断
df.filter(col('name').contains('Alice'))

# 分割
df = df.withColumn('name_parts', split(col('name'), ' '))
```

---

### 12. 读写文件

**Pandas:**
```python
# CSV
df = pd.read_csv('data.csv')
df.to_csv('output.csv', index=False)

# Parquet
df = pd.read_parquet('data.parquet')
df.to_parquet('output.parquet')
```

**Polars:**
```python
# CSV
df = pl.read_csv('data.csv')
df.write_csv('output.csv')

# Parquet - 推荐 ⭐
df = pl.read_parquet('data.parquet')
df.write_parquet('output.parquet')

# Lazy reading (大文件) ⭐⭐
df = pl.scan_parquet('data.parquet').collect()
```

**PySpark:**
```python
# CSV
df = spark.read.csv('data.csv', header=True, inferSchema=True)
df.write.mode('overwrite').csv('output.csv', header=True)

# Parquet
df = spark.read.parquet('data.parquet')
df.write.mode('overwrite').parquet('output.parquet')
```

---

## Polars 独特功能 🌟

### 1. Lazy Evaluation (延迟执行)

```python
# Lazy API - 构建查询计划，优化后执行
lazy_df = pl.scan_csv('big_file.csv')

result = (
    lazy_df
    .filter(pl.col('age') > 25)
    .select(['name', 'salary'])
    .groupby('name')
    .agg(pl.sum('salary'))
    .collect()  # 这里才真正执行
)

# 查看执行计划
print(lazy_df.explain())
```

### 2. 表达式强大的组合

```python
# 复杂的列操作
df = df.with_columns([
    # 条件逻辑
    pl.when(pl.col('age') > 30)
      .then(pl.col('salary') * 1.2)
      .otherwise(pl.col('salary'))
      .alias('adjusted_salary'),

    # 聚合表达式
    (pl.col('salary') - pl.col('salary').mean())
      .alias('salary_deviation'),

    # 窗口函数
    pl.col('salary').rank().over('department').alias('rank_in_dept')
])
```

### 3. Streaming Mode (流式处理)

```python
# 处理超大文件（不需要全部加载到内存）
result = (
    pl.scan_csv('huge_file.csv')
    .filter(pl.col('value') > 100)
    .groupby('category')
    .agg(pl.sum('value'))
    .collect(streaming=True)  # 流式处理
)
```

### 4. 并行处理

```python
# Polars 自动并行处理，无需配置
# 自动使用所有 CPU 核心

# 可以查看并行度
pl.threadpool_size()  # 查看线程池大小
```

---

## 性能对比基准测试 🏎️

### 场景：10GB CSV 文件，分组聚合

```python
# 数据集：1亿行，10 列

# Pandas
# 时间：~180 秒
# 内存：~15GB

# Polars (Eager)
# 时间：~25 秒  ← 快 7 倍！
# 内存：~8GB

# Polars (Lazy + Streaming)
# 时间：~30 秒
# 内存：~2GB  ← 内存效率极高！

# PySpark (单机)
# 时间：~40 秒
# 内存：~10GB
```

---

## 迁移指南

### Pandas → Polars

```python
# Pandas
import pandas as pd
df = pd.read_csv('data.csv')
result = df[df['age'] > 25].groupby('name')['salary'].mean()

# Polars（几乎一样的逻辑）
import polars as pl
df = pl.read_csv('data.csv')
result = df.filter(pl.col('age') > 25).groupby('name').agg(pl.mean('salary'))
```

**主要区别：**
1. `df['col']` → `pl.col('col')`
2. `df[df['col'] > 0]` → `df.filter(pl.col('col') > 0)`
3. `.apply()` → `.map()` 或表达式

---

## 何时使用哪个？决策树 🌲

```
开始
 │
 ├─ 数据量 < 5GB？
 │   ├─ Yes → 需要快速原型？
 │   │         ├─ Yes → Pandas (最简单)
 │   │         └─ No  → Polars (更快)
 │   │
 │   └─ No → 数据量 < 50GB？
 │             ├─ Yes → Polars ⭐ (单机最优)
 │             └─ No  → 数据量 > 100GB？
 │                       ├─ Yes → PySpark (分布式)
 │                       └─ No  → Polars Streaming
```

---

## 总结表格

| 场景 | 推荐工具 | 原因 |
|------|---------|------|
| **学习数据分析** | Pandas | 简单，资料多 |
| **生产环境（单机）** | Polars | 快，内存少 |
| **超大数据集（集群）** | PySpark | 分布式 |
| **快速原型** | Pandas | 熟悉，快速 |
| **性能优化** | Polars | 速度快 5-10x |
| **实时数据流** | PySpark Streaming | 企业级 |

---

## 学习建议 📚

### 推荐学习顺序：

1. **Pandas** (1-2 周) - 打基础
2. **Polars** (1 周) - 迁移很容易
3. **PySpark** (2-3 周) - 理解分布式

### 实践项目建议：

```python
# 初级：用 Pandas
# - 分析 CSV 文件（< 1GB）
# - 数据清洗和可视化

# 中级：用 Polars
# - 处理大文件（5-20GB）
# - 性能优化挑战

# 高级：用 PySpark
# - 分布式数据处理
# - 实时数据流处理
```

---

## 速查表

| 操作 | Pandas | Polars | PySpark |
|------|--------|--------|---------|
| **过滤** | `df[df['col'] > 0]` | `df.filter(pl.col('col') > 0)` | `df.filter(col('col') > 0)` |
| **选择** | `df[['a', 'b']]` | `df.select(['a', 'b'])` | `df.select('a', 'b')` |
| **添加列** | `df['new'] = df['old'] + 1` | `df.with_columns((pl.col('old') + 1).alias('new'))` | `df.withColumn('new', col('old') + 1)` |
| **分组** | `df.groupby('col').mean()` | `df.groupby('col').agg(pl.mean('*'))` | `df.groupBy('col').mean()` |
| **排序** | `df.sort_values('col')` | `df.sort('col')` | `df.orderBy('col')` |
| **去重** | `df.drop_duplicates()` | `df.unique()` | `df.dropDuplicates()` |

---

## 推荐资源

- **Pandas 文档**: https://pandas.pydata.org/
- **Polars 文档**: https://pola-rs.github.io/polars/
- **PySpark 文档**: https://spark.apache.org/docs/latest/api/python/

**Polars 是未来趋势！强烈推荐学习！** 🚀
