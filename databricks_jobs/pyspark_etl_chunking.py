#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##
# Description:-
#
#       Azure Databricks Distributed PySpark ETL & RAG Chunking Job.
#       Processes TB-scale multi-format enterprise files from ADLS Gen2 in parallel,
#       extracts structured text via worker UDFs, and persists chunks into Delta Lake.
#
##
# Development date    Developed by       Comments
# ----------------    ------------       ---------
# 16/01/2025          Saddam Khan        Initial implementation
# 20/02/2025          Saddam Khan        Added PySpark distributed document chunking and Delta Lake write
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, explode
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType
import os
import sys

# 1. Initialize Spark Session with Delta Lake & ADLS Gen2 connectors
spark = (
    SparkSession.builder.appName("Azure_Databricks_Enterprise_RAG_ETL")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)

# 2. Define schema for extracted chunks
chunk_schema = StructType([
    StructField("chunk_id", StringType(), False),
    StructField("text", StringType(), False),
    StructField("file_format", StringType(), False),
    StructField("section", StringType(), True),
    StructField("page", IntegerType(), True),
    StructField("allowed_roles", ArrayType(StringType()), True),
])

# 3. Distributed Parsing Worker Function
def parse_content_spark(filename: str, content_bytes: bytes):
    """
    Explanation: Distributed worker parser executing on Spark worker nodes to extract chunks
    :param  filename str: Path or name of source file
    :param  content_bytes bytes: Raw binary file payload
    :return chunks list: List of tuples matching chunk_schema
    """
    from parsers.factory import ParserFactory
    parser = ParserFactory.get_parser(filename)
    chunks = parser.parse_bytes(content_bytes, filename)
    return [
        (c.chunk_id, c.text, c.file_format, c.section, c.page, c.allowed_roles)
        for c in chunks
    ]

parse_udf = udf(parse_content_spark, ArrayType(chunk_schema))

def run_databricks_job(input_adls_path: str, delta_table: str):
    """
    Explanation: Distributed PySpark ETL workflow ingesting ADLS Gen2 files and writing to Delta Lake
    :param  input_adls_path str: ADLS Gen2 abfss URI pattern
    :param  delta_table str: Destination Delta table name
    :return None: Writes DataFrame to Delta Lake table
    """
    print(f"[*] Reading multi-terabyte binary files from ADLS Gen2: {input_adls_path}")

    # Read binary files in parallel across Spark cluster
    raw_df = spark.read.format("binaryFile").load(input_adls_path)

    # Distributed chunking
    parsed_df = (
        raw_df.withColumn("filename", col("path"))
        .withColumn("chunks", parse_udf(col("filename"), col("content")))
        .select(col("path"), explode("chunks").alias("chunk"))
        .select(
            col("path").alias("source_uri"),
            col("chunk.chunk_id").alias("chunk_id"),
            col("chunk.text").alias("text"),
            col("chunk.file_format").alias("file_format"),
            col("chunk.section").alias("section"),
            col("chunk.page").alias("page"),
            col("chunk.allowed_roles").alias("allowed_roles"),
        )
    )

    print(f"[*] Writing parsed enterprise chunks into Delta Lake: {delta_table}")
    (
        parsed_df.write.format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(delta_table)
    )

    print(f"[✓] Successfully processed and stored chunks in Delta Lake: {delta_table}")

if __name__ == "__main__":
    adls_path = sys.argv[1] if len(sys.argv) > 1 else "abfss://enterprise-knowledge@adlsgen2.dfs.core.windows.net/raw/*"
    out_table = sys.argv[2] if len(sys.argv) > 2 else "enterprise_rag.document_chunks_silver"
    # Note: Spark session execution is intended for Azure Databricks runtime
    print(f"Azure Databricks PySpark script prepared for ADLS: {adls_path} -> {out_table}")
