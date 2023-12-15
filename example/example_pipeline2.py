import os
import sys

from InterpretME import pipeline, plots


results = pipeline(
    path_config='./example_cvs_heart_failure_dataset.json',
    lime_results='./output/lime',
    survshap_results='./output/survshap',
    server_url='http://interpretmekg:8891/',
    username='dba',
    password='dba',
    survival=1
    )


