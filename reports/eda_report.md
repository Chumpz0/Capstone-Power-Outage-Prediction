# Exploratory Data Analysis Report: Storm-to-Outage

DSCI 4350 capstone, Georgia State University, Fall 2026. Prepared 2026-10-27 from existing notebook outputs, report tables, and figures.

**Template credit:** Report structure adapted from [SarahAA1's EDA Template](https://github.com/SarahAA1/EDA_Template/blob/main/EDA%20Template.ipynb). The project-specific analysis and figures come from this repository's notebooks. Sections are included where existing work supports them; no new analyses were run for this report.

## Executive Summary

The modeling dataset contains 271,644 county-by-storm-episode records for 2017-2025. Its target is whether at least 5% of a county's estimated customers were without power during a 24-hour window. Training prevalence is 14.13%; validation prevalence is 16.48%.

County outage history and pre-storm outage conditions have the strongest individual associations with the target. Storm family also separates risk: 62.5% of training episodes containing tropical reports are positive, compared with 7.7% containing winter reports. Missing intensity measurements often reflect storm type, and several feature pairs are strongly correlated.

The main limitations are uneven outage-observation coverage, whole-episode features that may not be available at storm start, and a target that includes counties already experiencing outages. Most feature distributions change little across splits, but the quality-score feature `dqi` shifts substantially. Results support retrospective model development; they do not establish readiness for live crew staging or prove that a storm caused an outage.

## Introduction

The project asks: **given a storm, what is the probability that a US county experiences a substantial power outage?** Utility crew staging motivates the question, while the current data use observed storm reports as a proxy for forecast information.

This report combines two completed analyses:

- [Notebook 03](../notebooks/03_eda.ipynb) explores cleaned outage and storm tables using training years only. Its main unit is an individual storm report paired with a county, and its initial outage threshold is 1%.
- [Notebook 06](../notebooks/06_ml_dataset_eda.ipynb) profiles the final modeling table, whose unit is county-by-episode and whose target threshold is 5%. Feature-target relationships use 2017-2023 training rows. Split comparisons use features, with previously reported validation prevalence included for context. Test outcomes are withheld.

Findings below refer to the final modeling table unless explicitly labeled as earlier event-level EDA. Percentages from these two populations should not be treated as interchangeable.

## Data Overview

### Dataset Information

**Dataset:** Storm-to-Outage, hosted in [DSCI-Capstone/county-outage-data](https://huggingface.co/datasets/DSCI-Capstone/county-outage-data). Sources are EAGLE-I outage readings, NOAA Storm Events, Census population, and geographic/coverage references. Full provenance and definitions are documented in the [project data card](dataset_card.md).

| Split | Years | Rows | Counties | Episodes | Positive rate |
|---|---|---:|---:|---:|---:|
| Training | 2017-2023 | 208,482 | 3,073 | 54,580 | 14.13% |
| Validation | 2024 | 31,064 | 3,023 | 7,734 | 16.48% |
| Test features | 2025 | 32,098 | 3,014 | 7,866 | Withheld |

Source: [dataset overview](ml_eda_tables/dataset_overview.csv). Scope is the lower 48 states and DC. Training and validation each have 46 columns: six identifiers, 36 candidate features, the target, and three retrospective analysis fields. Test features contain only the 42 identifier/feature columns.

### Data Loading and Cleaning

Existing notebooks use local source files where available and otherwise read from the project's private Hugging Face dataset. Preparation parses timestamps and identifiers, converts NOAA times to UTC, maps forecast-zone reports to counties, harmonizes Connecticut geography, and constructs customer denominators.

Null and zero outage readings are excluded from the positive-outage table, with observation status maintained separately. Missing hourly records are interpreted as zero only under the project's coverage assumptions. Readings above the denominator are capped, and hourly observations retain the worst snapshot within each hour.

The modeling-table preparation reduces 643,757 event-county records to 271,644 county-episode records after geographic filtering, exclusion of drought/heat triggers, episode aggregation, and observation-coverage filtering. See [row waterfall](model_table_tables/row_waterfall.csv) and [preparation summary](02_data_prep_summary.md).

Saved [ML dataset checks](ml_eda_tables/checks.csv) report unique county-episode keys, no missing training/validation targets, agreement between the label and the 5% threshold, non-overlapping split years, and no target in the test-feature file. These establish structural consistency rather than complete observation coverage or forecast-time validity.

## Data Summary and Descriptive Statistics

### Summary Statistics

The following are saved training statistics. Numeric summaries exclude missing values; outage shares are fractions between zero and one.

| Variable | Mean | Median | 25th-75th percentile | Maximum |
|---|---:|---:|---:|---:|
| Reports per county-episode | 1.88 | 1 | 1-2 | 83 |
| Maximum wind, knots | 54.03 | 52 | 50-56 | 173 |
| Maximum hail, inches | 1.29 | 1 | 1-1.5 | 6.42 |
| County episode duration, hours | 19.08 | 4 | 0.13-15 | 743.98 |
| Counties per episode | 9.70 | 7 | 3-13 | 96 |
| Historical county outage rate | 0.1343 | 0.1151 | 0.0641-0.1847 | 0.5832 |
| Peak share out, retrospective | 0.0335 | 0.0019 | 0.00005-0.0182 | 1.00 |

Source: [feature dictionary and statistics](ml_eda_tables/feature_dictionary.csv). The gaps between means and medians show long right tails in episode duration, report counts, and outage severity. The retrospective peak defines the outcome and must not be used as a predictor.

### Feature Distribution

![Existing distributions of selected features across training, validation, and test](figures/ml_eda/08_feature_distributions_by_split.png)

The selected distributions overlap substantially across splits. Most county-episodes have few reports, while duration and episode size have long tails. Recorded wind values cluster near 50 knots. Historical outage rate and county size vary widely across rows.

This existing plot excludes missing values and places values above the pooled 99th percentile in the final bin. That is a display choice, not evidence that extreme values were removed from the modeling table. Drift diagnostics below complement these visual comparisons.

## Missing Values Analysis

| Feature | Missing in training | Positive rate when missing | Positive rate when present |
|---|---:|---:|---:|
| Hurricane category | 99.92% | 14.07% | 90.18% |
| Maximum hail | 87.19% | 14.34% | 12.71% |
| Maximum wind | 59.05% | 8.63% | 22.06% |
| Pre-storm peak / preceding-hour outage share | 3.32% each | 11.62% | 14.21% |
| Log population | 0.36% | 11.26% | 14.14% |

Source: [missingness versus target](ml_eda_tables/missingness_vs_target.csv).

Wind, hail, and hurricane measurements are conditional on the kinds of events reported. Their missingness therefore contains information about episode composition. The small observed hurricane-category subset should not be treated as representative of all storms. Pre-storm features are unknown when the preceding day is not observed. Population gaps correspond to Connecticut's legacy county geography.

The modeling table retains these missing values. A missing intensity is not an observed zero. Any model-specific filling, indicators, or scaling must be learned from training data. Missing outage observations are a separate issue: the daily observation heuristic reduces ambiguity but cannot prove that every county was continuously monitored.

## Univariate Analysis

### Numeric Features

![Training distribution of retrospective peak outage share](figures/ml_eda/02_peak_share_distribution.png)

Training severity is highly skewed: 21.99% of county-episodes have a zero peak, 32.01% reach at least 1%, 14.13% reach 5%, 8.25% reach 10%, and 1.21% reach 50%. Among positives, median peak share is 11.87%. The threshold therefore identifies a minority class with substantial variation in severity. Source: [peak summary](ml_eda_tables/peak_summary.csv).

Long episode durations and high report counts also have strongly asymmetric distributions, as shown in the descriptive statistics and feature-distribution figure. These ranges should not be mistaken for data errors solely because they are far above the median.

### Categorical Features

Storm families are derived from report types and can coexist in an episode. Counts below describe training rows containing each family, not mutually exclusive categories.

| Family present | Training rows |
|---|---:|
| Thunderstorm wind | 59,883 |
| Winter | 55,210 |
| Flood | 29,586 |
| Hail | 26,701 |
| High wind | 26,049 |
| Tornado | 8,110 |
| Tropical | 2,029 |
| Ice storm | 1,574 |

Source: [storm-family table](ml_eda_tables/rate_by_storm_family.csv). The smaller tropical and ice-storm samples matter when interpreting their high positive rates. County-coded-only and zone-coded-only episodes are both common: 104,315 and 101,801 rows respectively; another 2,366 contain both report types. Source: [coding table](ml_eda_tables/rate_by_storm_coding.csv).

## Bivariate Analysis

### Numeric-Numeric Relationships

The saved Spearman analysis identifies strong associations among numerical inputs. County count and its log transform have identical rankings; customer count and population also track each other closely. The correlation section below reports the coefficients and existing heatmap.

### Numeric Features and the Binary Target

![Existing single-feature target-association chart](figures/ml_eda/03_univariate_signal.png)

County historical outage rate has a training ROC-AUC of 0.738 when used directly as a score, followed by historical mean peak at 0.732 and pre-storm peak at 0.668. Episode report count has ROC-AUC 0.645; thunderstorm-wind count has 0.638. Source: [single-feature signal](ml_eda_tables/univariate_signal.csv).

Each score uses rows where that feature is observed, so features with different missingness are evaluated on different subsets. The figure displays `max(AUC, 1-AUC)` for strength and uses color to indicate direction. For example, historical episode count has raw AUC 0.351 but displayed strength 0.649 because higher values are associated with fewer positives. These are training associations, not held-out model scores or estimates of causal importance.

![Training positive rates across existing feature bins](figures/ml_eda/04_rate_by_feature_bins.png)

The positive rate rises from 1.52% in the lowest county-history bin to 37.35% in the highest. The existing curves also show non-linear associations for county size and episode duration. These patterns support examining non-linear relationships rather than assuming a constant linear effect. Source: [feature-bin rates](ml_eda_tables/rate_by_feature_bins.csv).

### Categorical-Numeric Relationships

![Training outage rates by storm family, coding, and preceding-hour outage band](figures/ml_eda/05_family_coding_prestorm.png)

| Family present | Training positive rate |
|---|---:|
| Tropical | 62.49% |
| Ice storm | 33.16% |
| Tornado | 29.04% |
| Thunderstorm wind | 24.82% |
| High wind | 16.16% |
| Hail | 12.71% |
| Flood | 11.62% |
| Winter | 7.67% |

These differences describe associations between episode composition and a binary numerical outcome. They do not isolate the effect of a storm type from geography, intensity, or coverage. Source: [family rates](ml_eda_tables/rate_by_storm_family.csv).

County-coded-only episodes have a 17.86% positive rate, zone-coded-only episodes 9.89%, and mixed episodes 32.04%. Different event types use different location coding, so the contrast does not establish that coding itself causes different outcomes.

Pre-existing outages strongly separate outcomes: rows with less than 0.1% out in the preceding hour are positive 11.67% of the time, versus 89.37% for the 2,596 rows already at or above 5%. The current target includes these already-out rows and does not require a subsequent increase. Sources: [coding rates](ml_eda_tables/rate_by_storm_coding.csv) and [pre-storm bands](ml_eda_tables/rate_by_pre_storm_outage.csv).

## Feature Engineering

The existing modeling pipeline creates 36 candidate features:

| Group | Count | Existing construction and purpose |
|---|---:|---|
| Storm | 25 | Report-family counts, county/zone coding, maxima, duration, and episode extent summarize storm composition and scale |
| Time | 2 | Month and approximate local standard-time hour describe timing |
| County | 4 | Log customers, log population, regional DQI, and state coverage describe exposure and measurement context |
| Pre-storm | 2 | Peak share out over the preceding 24 hourly bins and share out in the preceding hour describe existing outages |
| History | 3 | County episode count, smoothed outage rate, and smoothed peak share describe training-period history |

History rates use a provisional prior weight of 20. Training rows exclude their own year from county-specific aggregates, while global priors use all training years; validation and test history use all training years. This is not a strictly past-only feature pipeline. Whole-episode storm summaries can also include information collected after the prediction time.

The pipeline excludes `target` and `retro_*` from predictors, but that exclusion alone does not establish forecast-time availability. See [notebook 04](../notebooks/04_modeling_table.ipynb) and the [data card](dataset_card.md) for exact construction and limitations.

## Outlier Detection

The completed audit and preparation identified implausible customer denominators and outage readings above those denominators. North Carolina's MCC estimate was approximately 0.17 customers per person, compared with roughly 0.5 nationally. The pipeline substitutes a Census-based estimate for all North Carolina counties, missing-MCC cases, and 21 other counties with implausible ratios.

County ratios below 0.1 or above 5 customers per person trigger substitution; these are provisional rules. Outage readings exceeding the selected denominator are capped, with the original count and a flag retained in the cleaned data. A spurious MCC "Grand Total" county row was also removed. These are documented data-quality corrections, not a blanket removal of extreme storms. Sources: [preparation summary](02_data_prep_summary.md) and [modeling-table summary](04_modeling_table_summary.md).

The final table still contains large values, including 173-knot reported wind, 6.42-inch hail, and nearly 744-hour county episodes. The saved summaries document these extrema but do not establish that each is erroneous. Plot clipping above the 99th percentile changes only the display.

## Correlation Analysis

![Training Spearman correlation matrix](figures/ml_eda/06_feature_correlation.png)

| Feature pair | Spearman correlation |
|---|---:|
| Episode counties / log episode counties | 1.000 |
| Log customers / log population | 0.980 |
| Zone-coded count / zone-coded share | 0.949 |
| Historical outage rate / historical mean peak | 0.944 |
| County-coded count / zone-coded share | -0.927 |

Source: [correlated pairs](ml_eda_tables/top_correlated_pairs.csv). The first pair is a deliberate monotonic transformation. Others indicate overlapping information or complementary coding patterns. Strong correlation can complicate interpretation of individual linear-model coefficients; it does not establish that either feature is invalid or justify automatic deletion. Correlation also does not establish causation.

## Custom Analyses

### Time, Season, and Region

![Target rates by year, month, and FEMA region](figures/ml_eda/01_target_by_year_month_region.png)

Training positive rates rise from 11.15% in 2017 to 16.85% in 2023, with fluctuations between years. Validation prevalence is 16.48%. EAGLE-I observation coverage also changes over time, so the increase cannot be attributed entirely to worsening weather or grid conditions. Source: [annual rates](ml_eda_tables/rows_by_year.csv).

Within training data, January's positive rate is 7.95% and June's is 20.11%. FEMA regions IV and VI are both near 20.6%, while region VIII is 4.84%. These patterns reflect the observed mix of storms, places, and measurement coverage. Source: [month and region rates](ml_eda_tables/target_by_month_region.csv).

### Feature Drift Across Splits

![Existing feature-drift chart](figures/ml_eda/07_feature_drift_psi.png)

The saved PSI table shows substantial DQI drift: 3.69 for validation and 3.75 for test. DQI is available only for 2018-2022 and mapped to neighboring years outside that range, making its distribution partly a product of the reference-year mapping. Customer coverage and historical episode count show smaller shifts, with PSI values around 0.04-0.06.

The chart excludes the extremely sparse hurricane-category feature and caps its displayed axis. The full table records hurricane-category PSI of 0.197 for validation and 8.064 for test; the feature is entirely missing in the test data. Its test value should not be interpreted as an ordinary comparison of observed category distributions. Missingness needs separate consideration because the PSI routine drops missing values. Source: [drift table](ml_eda_tables/feature_drift_psi.csv) and [feature dictionary](ml_eda_tables/feature_dictionary.csv).

### Earlier Event-Level EDA: Target Threshold and Window

The earlier EDA used training-year storm-report-by-county records before episode aggregation. Its comparison is the same county and hours one week earlier; this is a descriptive background comparison rather than a causal control group.

| Outage share reached | After storm report | One week earlier |
|---|---:|---:|
| At least 0.1% | 59% | 30% |
| At least 1% | 36% | 9% |
| At least 5% | 17% | 2% |
| At least 10% | 10% | 1% |

![Existing event-level severity-exceedance analysis](figures/06_outage_severity_exceedance.png)

The 5% threshold provides stronger relative separation from the background comparison than the original 1% threshold, at the cost of fewer positives. In the earlier timing analysis, median time to peak is four hours and 83% of examined peaks occur within 24 hours.

![Existing event-level time-to-peak analysis](figures/07_hours_to_peak.png)

These findings informed the chosen threshold and window; they do not make them operational standards. The final label uses hourly bins 0-23 starting at the hour containing `t0`, so the first bin can include observations before the exact report minute. Sources: [03 EDA summary](03_eda_summary.md) and [label construction](../notebooks/04_modeling_table.ipynb).

## Conclusions and Recommendations

The completed EDA supports a learnable association between storm characteristics, county history, pre-storm conditions, and outage occurrence. It also shows that the problem is shaped by observation coverage and label design. County-history signal, storm-family differences, and missingness patterns are useful model-development evidence; none independently proves storm causation or forecasting performance.

The evidence supports the following modeling guidance:

- Preserve time-based evaluation and account for within-episode dependence in uncertainty estimates.
- Treat missing storm measurements as informative; fit any preprocessing on training data.
- Account for non-linear relationships and correlated inputs when fitting and interpreting models.
- Treat DQI as a shifting measurement-context feature, and distinguish whole-episode retrospective inputs from features available at a real prediction time.
- Report calibration and subgroup behavior alongside ranking, and evaluate against the reproducible reference specified in [success criteria](success_criteria.md).

These recommendations follow the existing analyses. The report's figures and tables describe the current retrospective dataset, with the [data card](dataset_card.md) providing the fuller record of provenance and limitations.
