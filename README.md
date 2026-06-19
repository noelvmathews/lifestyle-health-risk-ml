# Lifestyle-Based Health Risk Analysis Using Machine Learning

An end-to-end data analysis and modeling project evaluating how individual daily habits (diet, sleep, physical activity) correlate with overall health risk categories. This project covers systematic Exploratory Data Analysis (EDA), unsupervised grouping via K-Means clustering, and supervised classification using Decision Trees.

## Project Structure
- `health_risk_analysis.py`: Modular Python script implementing the data workflow, statistical summaries, and model implementations.
- `dataset.csv`: Cleaned dataset containing lifestyle profiles of 200 individuals.

## Dataset Overview & Target Features
The analytical model processes 200 distinct samples across key structural variables:
- **Demographics & Biometrics:** Age, Gender (Categorical, encoded for training), and BMI (Body Mass Index).
- **Behavioral Patterns:** ExerciseHours (weekly), JunkFoodFreq (weekly), and SleepHours (daily).
- **Target Variable:** HealthRisk status, classified into distinct `Low`, `Medium`, and `High` categories.

---

## Technical Implementations & Section Insights

### Section 1: Exploratory Data Analysis (EDA)
- **Primary Driver:** Body Mass Index (BMI) demonstrated the strongest direct association with escalating health risk categories. The aggregate average BMI of the cohort falls squarely between `26 and 30`.
- **Risk Indicators:** Individuals categorized under **High Health Risk** show an active deficit in physical patterns, typically exercising for *less than 2 hours per week*, maintaining the highest average junk food intake, and logging the lowest average sleep duration.
- **Correlations:** A pronounced negative correlation was verified between `ExerciseHours` and `BMI`, illustrating how structured physical activity trends downward as body mass attributes increase.
- **Distribution:** Medium-risk subjects compose roughly `35%` of the total dataset population, while High-risk individuals represent the lowest overall sample frequency. Conversely, individuals maintaining a **Low Health Risk** profile consistently display healthy recovery metrics, averaging `7–8 hours` of daily sleep.

### Section 2: Unsupervised Learning (K-Means Clustering)
To segment the population into three distinct health profiles without pre-labeled targets ($k = 3$), the system evaluates spatial proximity across key wellness vectors:
- **Low-Risk Baseline:** The spatial centroid most representative of an optimal low-risk baseline maps to **Centroid: BMI = 22, Exercise = 5 hours/week**.
- **Centroid Optimization:** Algorithmic adjustment steps operate purely on spatial averages. For instance, a data cluster containing individual BMI values of `22, 24, 26, and 28` dynamically shifts its structural coordinate to an updated center of `25` following the centroid reassignment loop.

### Section 3: Supervised Machine Learning (Decision Tree)
A foundational Decision Tree classifier was modeled using `BMI` as the root node split condition ($\text{BMI} < 27$).



#### 1. Node Split Distribution
- **Yes Branch ($\text{BMI} < 27$):** Accounts for `117` individual samples.
  - Class Frequencies: `70` Low Risk, `47` Medium Risk, `0` High Risk.
  - Class Probabilities: $p_{\text{low}} = 0.598$, $p_{\text{medium}} = 0.402$, $p_{\text{high}} = 0.000$.
- **No Branch ($\text{BMI} \ge 27$):** Accounts for `83` individual samples.

#### 2. Impurity and Node Strategy
- Using the class distribution from the root split, the Gini Impurity of the child node ($\text{BMI} < 27$) was calculated using the equation:
  $$Gini = 1 - (p_{\text{low}}^2 + p_{\text{medium}}^2 + p_{\text{high}}^2)$$
  $$Gini = 1 - (0.598^2 + 0.402^2 + 0^2) = 0.48$$
- Because a Gini index of `0.48` indicates moderate impurity, it is determined that the node requires further feature-based splitting to maximize information gain. (A completely pure node containing a single target class defaults to an absolute index value of $Gini = 0$).
- When isolating target behavior against activity metrics ($\text{ExerciseHours} \le 5$) at the root node, the resultant "Yes" sub-branch yields an explicit trend where the majority of individuals fall into the `Medium Risk` spectrum.

#### 3. Generalization & Overfitting Analysis
During evaluation, the baseline model achieved a high training accuracy of `98%`, but plummeted to `25%` when subjected to independent, unseen test data. This significant variance indicates severe **overfitting**—where the tree memorizes the structural noise of the training dataset instead of extracting generalized boundaries. Mitigating this issue requires enforcing tree constraints, such as setting a `max_depth` limit or applying cost-complexity pruning.
