import os
import sys
import pickle
import pandas as pd
from InterpretME import pipeline, survshap



def main():
    path_config = './example_cvs_tlos_v1.json'
    results = pipeline(
        path_config=path_config,
        lime_results='./output/lime',
        survshap_results='./output/SurvSHAP',
        server_url='http://interpretmekg:8891/',
        username='dba',
        password='dba',
        survival=1
    )
    with open("./output/SurvSHAP/survshap_results.pkl", "rb") as file:
        pickle_expl_survshap= pickle.load(file)

    Ordering = survshap.calculate_feature_orderings (pickle_expl_survshap)
    Ordering.to_csv('./output/SurvSHAP/Ordering.csv', index=False)
    ranking_summary = survshap.create_ranking_summary(Ordering)
    ranking_summary.to_csv('./output/SurvSHAP/ranking_summary.csv', index=False)
    ranking_summary = pd.read_csv('./output/SurvSHAP/ranking_summary.csv')
    df= survshap.make_factors(ranking_summary)
    survshap.plotting_features_ranking_bars(df, title= 'SurvSHAP(t)', ytitle= 'Importance Ranking')
    survshap.plot_SurvSHAP_values_to_time("./output/SurvSHAP/SurvSHAP_files")
    survshap.plot_SurvSHAP_values_to_time_for_specific_file("./output/SurvSHAP/SurvSHAP_files", "patient_5")

if __name__ == '__main__':
    main()