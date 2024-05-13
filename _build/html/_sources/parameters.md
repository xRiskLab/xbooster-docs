# Parameters

<span style="font-family: Karma, sans-serif;">

This section describes the parameters and methods available in the `xbooster` library.

</span>

## `xbooster.constructor` - XGBoost Scorecard Constructor

### Description

<span style="font-family: Karma, sans-serif;">

A class for generating a scorecard from a trained XGBoost model. The methodology is inspired by the NVIDIA GTC Talk "Machine Learning in Retail Credit Risk" by Paul Edwards.

</span>

### Methods

<span style="font-family: Karma, sans-serif;">

1. `extract_leaf_weights() -> pd.DataFrame`:
   - Extracts the leaf weights from the booster's trees and returns a DataFrame.
   - **Returns**:
     - `pd.DataFrame`: DataFrame containing the extracted leaf weights.

2. `extract_decision_nodes() -> pd.DataFrame`:
   - Extracts the split (decision) nodes from the booster's trees and returns a DataFrame.
   - **Returns**:
     - `pd.DataFrame`: DataFrame containing the extracted split (decision) nodes.

3. `construct_scorecard() -> pd.DataFrame`:
   - Constructs a scorecard based on a booster.
   - **Returns**:
     - `pd.DataFrame`: The constructed scorecard.

4. `create_points(pdo=50, target_points=600, target_odds=19, precision_points=0, score_type='XAddEvidence') -> pd.DataFrame`:
   - Creates a points card from a scorecard.
   - **Parameters**:
     - `pdo` (int, optional): The points to double the odds. Default is 50.
     - `target_points` (int, optional): The standard scorecard points. Default is 600.
     - `target_odds` (int, optional): The standard scorecard odds. Default is 19.
     - `precision_points` (int, optional): The points decimal precision. Default is 0.
     - `score_type` (str, optional): The log-odds to use for the points card. Default is 'XAddEvidence'.
   - **Returns**:
     - `pd.DataFrame`: The points card.

5. `predict_score(X: pd.DataFrame) -> pd.Series`:
   - Predicts the score for a given dataset using the constructed scorecard.
   - **Parameters**:
     - `X` (`pd.DataFrame`): Features of the dataset.
   - **Returns**:
     - `pd.Series`: Predicted scores.

6. `sql_query` (property):
   - Property that returns the SQL query for deploying the scorecard.
   - **Returns**:
     - `str`: The SQL query for deploying the scorecard.

7. `generate_sql_query(table_name: str = "my_table") -> str`:
   - Converts a scorecard into an SQL format.
   - **Parameters**:
     - `table_name` (str): The name of the input table in SQL.
   - **Returns**:
     - `str`: The final SQL query for deploying the scorecard.

</span>

## `xbooster.explainer` - XGBoost Scorecard Explainer

<span style="font-family: Karma, sans-serif;">

This module provides functionalities for explaining XGBoost scorecards, including methods to extract split information, build interaction splits, visualize tree structures, plot feature importances, and more.

</span>

### Methods:

<span style="font-family: Karma, sans-serif;">

1. `extract_splits_info(features: str) -> list`:
   - Extracts split information from the DetailedSplit feature.
   - **Inputs**:
     - `features` (str): A string containing split information.
   - **Outputs**:
     - Returns a list of tuples containing split information (feature, sign, value).

2. `build_interactions_splits(scorecard_constructor: Optional[XGBScorecardConstructor] = None, dataframe: Optional[pd.DataFrame] = None) -> pd.DataFrame`:
   - Builds interaction splits from the XGBoost scorecard.
   - **Inputs**:
     - `scorecard_constructor` (Optional[XGBScorecardConstructor]): The XGBoost scorecard constructor.
     - `dataframe` (Optional[pd.DataFrame]): The dataframe containing split information.
   - **Outputs**:
     - Returns a pandas DataFrame containing interaction splits.

3. `split_and_count(scorecard_constructor: Optional[XGBScorecardConstructor] = None, dataframe: Optional[pd.DataFrame] = None, label_column: Optional[str] = None) -> pd.DataFrame`:
   - Splits the dataset and counts events for each split.
   - **Inputs**:
     - `scorecard_constructor` (Optional[XGBScorecardConstructor]): The XGBoost scorecard constructor.
     - `dataframe` (Optional[pd.DataFrame]): The dataframe containing features and labels.
     - `label_column` (Optional[str]): The label column in the dataframe.
   - **Outputs**:
     - Returns a pandas DataFrame containing split information and event counts.

4. `plot_importance(scorecard_constructor: Optional[XGBScorecardConstructor] = None, metric: str = "Likelihood", normalize: bool = True, method: Optional[str] = None, dataframe: Optional[pd.DataFrame] = None, **kwargs: Any) -> None`:
   - Plots the importance of features based on the XGBoost scorecard.
   - **Inputs**:
     - `scorecard_constructor` (Optional[XGBScorecardConstructor]): The XGBoost scorecard constructor.
     - `metric` (str): Metric to plot ("Likelihood" (default), "NegLogLikelihood", "IV", or "Points").
     - `normalize` (bool): Whether to normalize the importance values (default: True).
     - `method` (Optional[str]): The method to use for plotting the importance ("global" or "local").
     - `dataframe` (Optional[pd.DataFrame]): The dataframe containing features and labels.
     - `fontfamily` (str): The font family to use for the plot (default: "Monospace").
     - `fontsize` (int): The font size to use for the plot (default: 12).
     - `dpi` (int): The DPI of the plot (default: 100).
     - `title` (str): The title of the plot (default: "Feature Importance").
     - `**kwargs` (Any): Additional Matplotlib parameters.

5. `plot_score_distribution(y_true: pd.Series = None, y_pred: pd.Series = None, n_bins: int = 25, scorecard_constructor: Optional[XGBScorecardConstructor] = None, **kwargs: Any)`:
   - Plots the distribution of predicted scores based on actual labels.
   - **Inputs**:
     - `y_true` (pd.Series): The true labels.
     - `y_pred` (pd.Series): The predicted labels.
     - `n_bins` (int): Number of bins for histogram (default: 25).
     - `scorecard_constructor` (Optional[XGBScorecardConstructor]): The XGBoost scorecard constructor.
     - `**kwargs` (Any): Additional Matplotlib parameters.

6. `plot_local_importance(scorecard_constructor: Optional[XGBScorecardConstructor] = None, metric: str = "Likelihood", normalize: bool = True, dataframe: Optional[pd.DataFrame] = None, **kwargs: Any) -> None`:
   - Plots the local importance of features based on the XGBoost scorecard.
   - **Inputs**:
     - `scorecard_constructor` (Optional[XGBScorecardConstructor]): The XGBoost scorecard constructor.
     - `metric` (str): Metric to plot ("Likelihood" (default), "NegLogLikelihood", "IV", or "Points").
     - `normalize` (bool): Whether to normalize the importance values (default: True).
     - `dataframe` (Optional[pd.DataFrame]): The dataframe containing features and labels.
     - `fontfamily` (str): The font family to use for the plot (default: "Arial").
     - `fontsize` (int): The font size to use for the plot (default: 12).
     - `boxstyle` (str): The rounding box style to use for the plot (default: "round").
     - `title` (str): The title of the plot (default: "Local Feature Importance").
     - `**kwargs` (Any): Additional parameters to pass to the matplotlib function.

7. `plot_tree(tree_index: int, scorecard_constructor: Optional[XGBScorecardConstructor] = None, show_info: bool = True) -> None`:
   - Plots the tree structure.
   - **Inputs**:
     - `tree_index` (int): Index of the tree to plot.
     - `scorecard_constructor` (Optional[XGBScorecardConstructor]): The XGBoost scorecard constructor.
     - `show_info` (bool): Whether to show additional information (default: True).
     - `**kwargs` (Any): Additional Matplotlib parameters.
</span>