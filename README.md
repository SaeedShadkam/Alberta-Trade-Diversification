# Neural Network-Based Trade Flow Prediction Model

## Introduction

This project is an advanced predictive analytics system that leverages deep learning to forecast international trade flows with a specific focus on Alberta's export diversification opportunities. Built upon the theoretical foundation of **gravity models in international trade**, the system addresses a critical challenge in international trade: accurately predicting bilateral trade volumes between economies based on comprehensive economic, geographic, and policy factors.

The model serves as a strategic tool for trade policy analysis and export market identification, providing quantitative insights into potential trade relationships and market opportunities. By analyzing historical trade patterns alongside macroeconomic indicators, the system generates data-driven predictions that can inform investment decisions, policy formulation, and market entry strategies.

### Project Objectives
- Introduce a novel approach to international trade flow modeling by integrating advanced embedding methods with gravity model theory
- Develop a deep neural network architecture that captures complex non-linear relationships in bilateral trade dynamics
- Pioneer the application of learnable commodity embeddings to model latent economic relationships between product categories
- Identify and quantify untapped export opportunities for Alberta's economy across diverse international markets
- Provide data-driven insights for trade diversification strategies and evidence-based policy formulation

## Methodology

The project implements a sophisticated neural network architecture using **PyTorch**, with preprocessing pipelines developed in both **Python** and **Julia**. The model integrates multiple  data sources to capture the complex dynamics of international trade.

### Data Sources
- **CEPII BACI Database**: Comprehensive bilateral trade data providing detailed import/export flows
- **UN Comtrade**: International trade statistics and commodity classifications (HS codes)
- **Statistics Canada**: Provincial export data
- **World Bank**: Macroeconomic indicators and development metrics
- **Global Trade Alert**: Trade policy interventions and regulatory measures

### Theoretical Framework
The model extends the classical **gravity model of international trade**, which posits that trade flows between countries are proportional to their economic sizes and inversely proportional to the distance between them. While traditional gravity models rely on linear relationships and limited variables, this neural network implementation captures complex non-linear interactions among:
- Economic mass variables (GDP, market size)
- Trade costs (distance, tariffs, policy barriers)
- Comparative advantage indicators
- Product-specific characteristics through learned embeddings

### Model Architecture
The neural network employs a ** architecture** that processes four distinct feature categories through specialized sub-networks:
- **Trade-specific features**: Price indicators, trade volumes, and complementarity measures
- **Exporter characteristics**: Economic performance, export concentration, and competitiveness metrics
- **Importer characteristics**: Market size, tariff structures, and import demand patterns
- **Bilateral factors**: Geographic distance, contiguity, and historical trade relationships

The model uses **K-fold cross-validation** (5 folds) with 200 epochs of training, achieving robust performance across multiple evaluation metrics including MSE, R², and MAPE.

## Novel Contribution: HS Code Embedding Approach

This project introduces an **innovative embedding methodology** for commodity classification that represents a significant advancement over traditional categorical encoding approaches in trade flow modeling.

### Technical Innovation
Rather than treating Harmonized System (HS) commodity codes as discrete categorical variables, the model implements **learnable embedding vectors** that capture latent economic relationships between different product categories. Each HS4 code (4-digit commodity classification) is mapped to a **32-dimensional dense vector** through a neural embedding layer.

![Embedding](Figs/Embedding.png)


### Economic Significance
The embedding approach allows the model to:
1. **Discover Product Similarities**: Automatically identify economically related commodities (e.g., different petroleum products, various agricultural goods)
2. **Capture Substitution Effects**: Model how trade patterns in similar products influence each other
3. **Reduce Dimensionality**: Efficiently handle thousands of commodity codes without sparse categorical variables
4. **Enable Transfer Learning**: Apply learned commodity relationships across different country pairs and time periods


### Practical Applications
The trained model generates **Alberta-specific trade predictions** for 2024, providing:
- Commodity-level export forecasts across 30+ international markets
- Currency-adjusted predictions (USD to CAD conversion using real-time exchange rates)
- Interactive visualizations comparing predicted vs. actual trade flows
- Identification of underexploited market opportunities and export potential


## Feature Engineering

The model incorporates **13 core trade features** and **11 macroeconomic indicators**, all calculated as 3-year moving averages to reduce volatility and capture trend dynamics:

### Trade-Specific Features (13 variables)
- **MA_AvgUnitPrice**: Average unit price of bilateral trade flows
- **MA_AvgUnitPriceFlags**: Data quality indicators for price imputation
- **MA_AvgUnitPriceofImporterFromWorld**: Global import price benchmarks for importers
- **MA_AvgUnitPriceofExporterToWorld**: Global export price benchmarks for exporters
- **MA_TotalImportofCmdbyReporter**: Total import demand by commodity and country
- **MA_TotalExportofCmdbyPartner**: Total export supply by commodity and country
- **MA_Trade_Complementarity**: index measuring export-import matching
- **MA_Partner_Revealed_Comparative_Advantage**: Export specialization indicators
- **MA_Liberalising**: Count of trade-liberalizing policy interventions
- **MA_Harmful**: Count of trade-restricting policy interventions
- **Covid**: Binary indicator for pandemic impact (2020-2023)

### Macroeconomic Indicators (11 variables)
**Exporter Characteristics (4 variables):**
- **MA_Theil_Exporter_Concentration**: Export diversification index
- **MA_GDPPerCapita_exporter**: Economic development level
- **MA_GeopoliticalIndex_exporter**: Political risk and stability measure
- **MA_ConsumerPriceIndex_exporter**: Inflation and monetary conditions

**Importer Characteristics (5 variables):**
- **MA_Theil_Importer_Concentration**: Import diversification index
- **MA_GDPPerCapita_importer**: Market size and purchasing power
- **MA_TariffRatesAllProductsWeightedAverage_importer**: Trade protection levels
- **MA_GeopoliticalIndex_importer**: Political and institutional quality
- **MA_ConsumerPriceIndex_importer**: Price level and competitiveness

**Bilateral Geographic Factors (2 variables):**
- **MA_contig**: Geographic contiguity indicator
- **MA_dist**: Weighted distance between economic centers

## Conclusion

This neural network-based gravity model demonstrates the substantial value of applying modern machine learning techniques to international trade analysis. The project successfully combines domain expertise in trade economics with cutting-edge neural network architectures to create a practical tool for trade policy and business strategy.

### Key Achievements
1. **Methodological Innovation**: Introduction of commodity embedding techniques that capture economic relationships between products
2. **Comprehensive Feature Integration**: Successful integration of trade, macroeconomic, and policy variables into a unified predictive framework
3. **Practical Application**: Development of Alberta-specific trade forecasting capabilities with real-world policy relevance
4. **Robust Performance**: Achievement of high predictive accuracy across diverse commodity sectors and international markets

### Strategic Impact
The model provides quantitative evidence for trade diversification strategies, enabling data-driven decisions in export promotion, trade agreement negotiations, and international market development. By identifying specific commodity-country combinations with high export potential, the system supports targeted investment in trade infrastructure and market development initiatives.

This work contributes to the growing field of **computational (trade) economics** and demonstrates how advanced analytics can enhance traditional trade forecasting methodologies. The embedding approach, in particular, offers a replicable framework for commodity relationship modeling that could be applied to other regional economies and trade forecasting challenges.

---
*Developed using PyTorch, Julia, and extensive international trade databases. Model training performed on GPU-accelerated infrastructure with comprehensive cross-validation protocols.*
