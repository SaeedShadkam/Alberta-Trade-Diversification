import os
import json
from pyspark.sql.types import FloatType
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from my_spark_functions import save_spark_df, read_spark_df
from Process_CEPII_tools import calculate_HS4_basket_distance, RCA, calculate_theil_exporter_concentration, calculate_theil_importer_concentration, calculate_trade_complementarity


# Setting up the directories, sparks, root_path
cwd = os.getcwd()
parent_directory = os.path.dirname(os.path.dirname(cwd))
with open(f"{parent_directory}/model_parameters.json", "r") as f:
    model_parameters = json.load(f)

# key vault
storage_account = "jetitih0018dlad91"
scope_name = "storage-account-kv"
secret_name = "storage-to-databricks-connect-kv"
secret_keys = dbutils.secrets.get(scope=scope_name, key=secret_name)

spark.conf.set(
    f"fs.azure.account.key.{storage_account}.dfs.core.windows.net",
    secret_keys
)

root_path = f"{model_parameters['root_path']}"
root_path = root_path + "/TariffData"

RegionCode = spark.read \
    .option("header", "true") \
    .csv(root_path)
    

cols_to_keep = ["Reporter_ISO_N", "Year", "ProductCode", "Partner", "SimpleAverage", "Nbr_Pref_Lines", "Nbr_MFN_Lines"]
PrefTariff = spark.read \
    .option("header", "true") \
    .csv(f'{root_path}/TariffDataPrefWITS1990-2023') \
    .select(cols_to_keep)


cols_to_keep = ["Reporter_ISO_N", "Year", "ProductCode", "SimpleAverage"]
MfnTariff = spark.read \
    .option("header", "true") \
    .csv(f'{root_path}/TariffDataWITS1990-2023') \
    .select(cols_to_keep)
    
PrefTariff = PrefTariff.join(
    RegionCode.select("RegionCode", "PartnerCode"),
    PrefTariff["Partner"] == RegionCode["RegionCode"], 
    how="left"
)
PrefTariff = PrefTariff.withColumn(
    "Partner",
    F.when(F.col("PartnerCode").isNotNull(), F.col("PartnerCode"))
     .otherwise(F.col("Partner"))
)
PrefTariff = PrefTariff.drop("RegionCode", "PartnerCode")
PrefTariff = PrefTariff.withColumnRenamed('SimpleAverage', 'PrefTariff')
MfnTariff = MfnTariff.withColumnRenamed('SimpleAverage', 'MFNTariff')
TariffTable = PrefTariff.join(
    MfnTariff,
    on=["Reporter_ISO_N", "Year", "ProductCode"]
)

TariffTable = TariffTable.filter(F.col("year").cast("int") >= model_parameters['Start_year'])




TariffTable = TariffTable.withColumn(
                                        "TariffRate",
                                        (
                                            F.col("PrefTariff").cast(FloatType()) * F.col("Nbr_Pref_Lines").cast(FloatType()) +
                                            F.col("MFNTariff").cast(FloatType()) * F.col("Nbr_MFN_Lines").cast(FloatType())
                                        ) / (
                                            F.col("Nbr_Pref_Lines").cast(FloatType()) + F.col("Nbr_MFN_Lines").cast(FloatType())
                                        )
                                    )

TariffTable.select("Reporter_ISO_N", "Year", "ProductCode", "Partner", "TariffRate", "MFNTariff" )
save_spark_df(TariffTable, f"{model_parameters['root_path']}", "TariffDataProcessed", spark)
