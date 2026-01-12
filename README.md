# PySpark 快速上手教程

## 简介
这是一个 PySpark 入门教程，涵盖了数据仓库建模（星型模式）和常用操作。

## 内容包括
- ✅ PySpark 环境配置
- ✅ Spark Session 创建
- ✅ 星型模式（Star Schema）数据仓库示例
- ✅ 多维度 JOIN 查询
- ✅ 聚合分析
- ✅ SQL 查询
- ✅ 窗口函数
- ✅ WordCount 经典案例

## 运行环境
- Python 3.9+
- PySpark 4.0.1
- Jupyter Notebook

## 如何运行
```bash
# 安装依赖
pip install pyspark pandas numpy

# 启动 Jupyter Notebook
jupyter notebook PySpark_Setup.ipynb
```

## 学习要点
1. **星型模式**：1 个事实表 + N 个维度表的数据仓库建模
2. **Shuffle 操作**：JOIN、groupBy、orderBy 会触发数据重分区
3. **本地模式**：`local[*]` 适合学习和小数据测试

## 参考资源
- [PySpark 官方文档](https://spark.apache.org/docs/latest/api/python/)
- [Spark SQL 指南](https://spark.apache.org/docs/latest/sql-programming-guide.html)
