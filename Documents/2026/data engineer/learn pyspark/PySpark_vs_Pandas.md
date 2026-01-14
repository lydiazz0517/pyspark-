# PySpark vs Pandas 语法对比

## 为什么要用 PySpark？

| 特性 | Pandas | PySpark |
|------|--------|---------|
| **数据规模** | 单机内存限制（通常 < 10GB） | 分布式处理，TB 级数据 |
| **处理速度** | 单线程/多线程 | 分布式并行处理 |
| **适用场景** | 小规模数据分析、原型开发 | 大数据处理、生产环境 |
| **学习曲线** | 简单直观 | 稍复杂，但概念相似 |

---

## 常用操作对比

### 1. 创建 DataFrame

**Pandas:**
```python
import pandas as pd

df = pd.DataFrame({
    'id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35]
})
```

**PySpark:**
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

df = spark.createDataFrame([
    (1, 'Alice', 25),
    (2, 'Bob', 30),
    (3, 'Charlie', 35)
], ['id', 'name', 'age'])
```

---

### 2. 查看数据

**Pandas:**
```python
df.head()        # 查看前 5 行
df.head(10)      # 查看前 10 行
df.tail()        # 查看后 5 行
df.info()        # 数据信息
df.describe()    # 统计描述
```

**PySpark:**
```python
df.show()        # 默认显示 20 行
df.show(10)      # 显示 10 行
df.take(5)       # 获取前 5 行（返回 Row 对象）
df.printSchema() # 查看结构
df.describe().show()  # 统计描述
```

---

### 3. 过滤数据 ⭐

**Pandas:**
```python
# 方法 1：布尔索引（最常用）
df_filtered = df[df['age'] > 25]

# 方法 2：多条件
df_filtered = df[(df['age'] > 25) & (df['name'] == 'Alice')]

# 方法 3：query()
df_filtered = df.query('age > 25')
```

**PySpark:**
```python
from pyspark.sql.functions import col

# 方法 1：filter() + col()
df_filtered = df.filter(col('age') > 25)

# 方法 2：多条件
df_filtered = df.filter((col('age') > 25) & (col('name') == 'Alice'))

# 方法 3：SQL 字符串
df_filtered = df.filter("age > 25")
```

---

### 4. 选择列

**Pandas:**
```python
# 单列
df['name']
df.name  # 不推荐

# 多列
df[['name', 'age']]
```

**PySpark:**
```python
# 单列
df.select('name')
df.select(col('name'))

# 多列
df.select('name', 'age')
df.select(col('name'), col('age'))
```

---

### 5. 添加新列

**Pandas:**
```python
# 方法 1
df['age_plus_10'] = df['age'] + 10

# 方法 2：assign()
df = df.assign(age_plus_10=df['age'] + 10)
```

**PySpark:**
```python
# 使用 withColumn()
df = df.withColumn('age_plus_10', col('age') + 10)
```

---

### 6. 删除列

**Pandas:**
```python
# 方法 1
df = df.drop('age', axis=1)

# 方法 2：drop() 不改变原 df
df_new = df.drop(columns=['age'])
```

**PySpark:**
```python
df = df.drop('age')
```

---

### 7. 重命名列

**Pandas:**
```python
df = df.rename(columns={'name': 'full_name'})
```

**PySpark:**
```python
df = df.withColumnRenamed('name', 'full_name')
```

---

### 8. 分组聚合

**Pandas:**
```python
# 单个聚合
df.groupby('name')['age'].mean()

# 多个聚合
df.groupby('name').agg({
    'age': 'mean',
    'salary': ['sum', 'count']
})
```

**PySpark:**
```python
from pyspark.sql.functions import mean, sum, count

# 单个聚合
df.groupBy('name').agg(mean('age'))

# 多个聚合
df.groupBy('name').agg(
    mean('age').alias('avg_age'),
    sum('salary').alias('total_salary'),
    count('*').alias('count')
)
```

---

### 9. 排序

**Pandas:**
```python
# 升序
df.sort_values('age')

# 降序
df.sort_values('age', ascending=False)

# 多列
df.sort_values(['name', 'age'], ascending=[True, False])
```

**PySpark:**
```python
from pyspark.sql.functions import desc, asc

# 升序
df.orderBy('age')
df.orderBy(asc('age'))

# 降序
df.orderBy(desc('age'))

# 多列
df.orderBy('name', desc('age'))
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

### 11. 去重

**Pandas:**
```python
df.drop_duplicates()
df.drop_duplicates(subset=['name'])
```

**PySpark:**
```python
df.dropDuplicates()
df.dropDuplicates(['name'])
```

---

### 12. JOIN 操作

**Pandas:**
```python
# Inner join
pd.merge(df1, df2, on='id')

# Left join
pd.merge(df1, df2, on='id', how='left')

# Multiple keys
pd.merge(df1, df2, on=['id', 'name'])
```

**PySpark:**
```python
# Inner join
df1.join(df2, 'id')
df1.join(df2, df1.id == df2.id)

# Left join
df1.join(df2, 'id', 'left')

# Multiple keys
df1.join(df2, ['id', 'name'])
```

---

### 13. 条件逻辑

**Pandas:**
```python
import numpy as np

# 简单条件
df['category'] = np.where(df['age'] > 30, 'Senior', 'Junior')

# 多条件
df['category'] = np.select(
    [df['age'] < 25, df['age'] < 35, df['age'] >= 35],
    ['Junior', 'Mid', 'Senior']
)
```

**PySpark:**
```python
from pyspark.sql.functions import when

# 简单条件
df = df.withColumn('category',
    when(col('age') > 30, 'Senior').otherwise('Junior')
)

# 多条件
df = df.withColumn('category',
    when(col('age') < 25, 'Junior')
    .when(col('age') < 35, 'Mid')
    .otherwise('Senior')
)
```

---

### 14. 字符串操作

**Pandas:**
```python
# 转大写
df['name'] = df['name'].str.upper()

# 分割
df['email_parts'] = df['email'].str.split('@')

# 替换
df['email'] = df['email'].str.replace('.com', '.org')
```

**PySpark:**
```python
from pyspark.sql.functions import upper, split, regexp_replace

# 转大写
df = df.withColumn('name', upper(col('name')))

# 分割
df = df.withColumn('email_parts', split(col('email'), '@'))

# 替换
df = df.withColumn('email',
    regexp_replace(col('email'), '.com', '.org')
)
```

---

### 15. 读写文件

**Pandas:**
```python
# 读取 CSV
df = pd.read_csv('data.csv')

# 写入 CSV
df.to_csv('output.csv', index=False)

# 读取 Parquet
df = pd.read_parquet('data.parquet')

# 写入 Parquet
df.to_parquet('output.parquet')
```

**PySpark:**
```python
# 读取 CSV
df = spark.read.csv('data.csv', header=True, inferSchema=True)

# 写入 CSV
df.write.mode('overwrite').csv('output.csv', header=True)

# 读取 Parquet
df = spark.read.parquet('data.parquet')

# 写入 Parquet
df.write.mode('overwrite').parquet('output.parquet')
```

---

## 关键区别总结

### 1. **执行模式**

**Pandas:**
- 立即执行（Eager Execution）
- 每个操作立即计算结果

**PySpark:**
- 延迟执行（Lazy Evaluation）
- 构建执行计划，调用 action 时才执行
- Action 操作：`show()`, `count()`, `collect()`, `write()`

### 2. **返回类型**

**Pandas:**
```python
df['age'].mean()  # 返回标量值 (float)
```

**PySpark:**
```python
df.select(mean('age')).show()  # 需要 show() 才能看到结果
df.agg({'age': 'mean'}).collect()[0][0]  # 获取标量值
```

### 3. **索引**

**Pandas:**
- 有行索引（index）
```python
df.loc[0]  # 通过索引访问
df.iloc[0]  # 通过位置访问
```

**PySpark:**
- 没有行索引
- 需要用 `monotonically_increasing_id()` 创建索引

### 4. **内存管理**

**Pandas:**
- 数据必须全部加载到内存
- 内存不足会崩溃

**PySpark:**
- 分布式存储，不受单机内存限制
- 自动处理大于内存的数据

---

## 何时使用哪个？

### 使用 Pandas：
✅ 数据 < 5GB
✅ 单机处理足够
✅ 快速原型开发
✅ 交互式数据探索
✅ 配合 Jupyter Notebook

### 使用 PySpark：
✅ 数据 > 10GB
✅ 需要分布式处理
✅ 生产环境 ETL
✅ 与 Hadoop/HDFS 集成
✅ 实时数据处理（Spark Streaming）

---

## 转换技巧

### Pandas → PySpark
```python
# Pandas
import pandas as pd
pandas_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

# 转换为 PySpark
spark_df = spark.createDataFrame(pandas_df)
```

### PySpark → Pandas
```python
# PySpark
spark_df = spark.createDataFrame([(1, 3), (2, 4)], ['a', 'b'])

# 转换为 Pandas（小心内存！）
pandas_df = spark_df.toPandas()
```

⚠️ **警告：** `toPandas()` 会将所有数据收集到 driver，大数据集会导致内存溢出！

---

## 练习：过滤负数金额

### Pandas 版本
```python
import pandas as pd

orders_df = pd.DataFrame({
    'order_id': ['ORD001', 'ORD002', 'ORD003'],
    'amount': [100.0, -50.0, 200.0]
})

# 过滤负数
orders_filtered = orders_df[orders_df['amount'] >= 0]
print(orders_filtered)
```

### PySpark 版本
```python
from pyspark.sql.functions import col

orders_df = spark.createDataFrame([
    ('ORD001', 100.0),
    ('ORD002', -50.0),
    ('ORD003', 200.0)
], ['order_id', 'amount'])

# 过滤负数
orders_filtered = orders_df.filter(col('amount') >= 0)
orders_filtered.show()
```

---

## 速查表

| 操作 | Pandas | PySpark |
|------|--------|---------|
| 过滤 | `df[df['col'] > 0]` | `df.filter(col('col') > 0)` |
| 选择列 | `df[['col1', 'col2']]` | `df.select('col1', 'col2')` |
| 添加列 | `df['new'] = df['old'] + 1` | `df.withColumn('new', col('old') + 1)` |
| 分组 | `df.groupby('col').mean()` | `df.groupBy('col').mean()` |
| 排序 | `df.sort_values('col')` | `df.orderBy('col')` |
| 去重 | `df.drop_duplicates()` | `df.dropDuplicates()` |
| JOIN | `pd.merge(df1, df2)` | `df1.join(df2)` |

---

## 推荐学习路径

1. 先学 Pandas（更简单）
2. 理解数据处理的基本概念
3. 学习 PySpark（迁移概念）
4. 在实际项目中根据数据规模选择工具
