import numpy as np
import pandas as pd
import os
from dataloader import *
from judgebioextractor import *
from bioanalyzer import analyzetext


def main():
    df = loader()

    # bioAndgrantrate = judgeAnalyzer(df)
    # csvwriter(bioAndgrantrate)

    analyzetext(df)

if __name__ == "__main__":
    main()
