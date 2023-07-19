import pandas as pd
import numpy as np
import abc
import utils
import config as cfg
from itertools import product
from metric_builder import Metric, CalculateMetric
from stattests import TTestFromStats, calculate_statistics, calculate_linearization
from stattests import MannWhitney, ProportionsZtest
from tqdm import tqdm


class Report:
    def __init__(self, report):
        self.report = report


class BuildMetricReport:
    def __call__(self, calculated_metric, metric_items, mc_estimator=None) -> Report:
        # ttest = TTestFromStats()
        estimators_dict = {
            "t_test": TTestFromStats(),
            "mann_whitney": MannWhitney(),
            "prop_test": ProportionsZtest()
        }

        if mc_estimator:
            estimator = estimators_dict[mc_estimator]
            cfg.logger.info(f"{mc_estimator}")
        else:
            estimator = estimators_dict[metric_items.estimator]
            cfg.logger.info(f"{metric_items.name}")

        # df_ = calculate_linearization(calculated_metric)
        # stats = calculate_statistics(df_, metric_items.type)
        # criteria_res = ttest(stats)
        stats = calculate_statistics(calculated_metric, metric_items.type)
        criteria_res = estimator(stats)

        report_items = pd.DataFrame({
            "metric_name": metric_items.name,
            "mean_0": stats.mean_0,
            "mean_1": stats.mean_1,
            "var_0": stats.var_0,
            "var_1": stats.var_1,
            "delta": stats.mean_1 - stats.mean_0,
            "lift": (stats.mean_1 - stats.mean_0) / stats.mean_0,
            "pvalue": criteria_res.pvalue,
            "statistic": criteria_res.statistic
        }, index=[0])

        return Report(report_items)


def build_experiment_report(df, metric_config, mc_config=None):
    build_metric_report = BuildMetricReport()
    reports = []
    _reports = []
    N = 10
    ALPHA = 0.05

    if mc_config is not None:
        for metric_params in tqdm(metric_config):
            metric_parsed = Metric(metric_params)

            lifts = mc_config.get("lifts", "default_value")

            # print(lifts.get('start'), lifts.get('end'), lifts.get('by'))

            for lift in tqdm(np.arange(lifts.get('start'), lifts.get('end'), step=lifts.get('by'))):
                for i in range(N):
                    for mc_estimator in mc_config.get('estimators'):
                        df_ = df.copy()
                        df_[cfg.VARIANT_COL] = np.random.choice(2, len(df))
                        df_.loc[df_[cfg.VARIANT_COL] == 1, metric_parsed.numerator_aggregation_field] = (
                                df_.loc[df_[cfg.VARIANT_COL] == 1, metric_parsed.numerator_aggregation_field] * lift
                        )

                        for metric_params in metric_config:
                            metric_parsed = Metric(metric_params)
                            calculated_metric = CalculateMetric(metric_parsed)(df_)
                            metric_report = build_metric_report(calculated_metric, metric_parsed, mc_estimator)
                            _reports.append(metric_report.report)

                        report = pd.concat(_reports)

                        report['mc_estimator'] = mc_estimator
                        report['is_reject'] = report['pvalue'] < ALPHA
                        report['real_lift'] = lift
                        report['iter'] = i
                        # print(report.columns)
                        # print(report.head())

                        reports.append(Report(report).report)

            # print(reports.columns)
            rep_out = pd.concat(reports)

            rep_gr = rep_out.groupby([
                'metric_name', 'mc_estimator', 'real_lift',
                'mean_0', 'mean_1', 'var_0', 'var_1', 'delta'
            ]).agg({'is_reject': np.mean, 'pvalue': np.mean}).reset_index()

            return rep_gr

    for metric_params in metric_config:
        metric_parsed = Metric(metric_params)
        calculated_metric = CalculateMetric(metric_parsed)(df)
        metric_report = build_metric_report(calculated_metric, metric_parsed)
        reports.append(metric_report.report)

    return pd.concat(reports)
