from Nifty_Tracker import Tracker
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm

base_dir = 'QuarterlyResults'
tracker = Tracker('.ini')

nifty_500 =  pd.read_csv('data/tracking_list.csv')

for resFile in tqdm(os.listdir(base_dir)[8:], desc='Processing Quarter'):
    if not resFile.endswith('announcements.csv'):
        continue
    print("Processing", resFile)
    results_df = pd.read_csv(os.path.join(base_dir, resFile))
    for i, ann in tqdm(results_df.iterrows(), desc='Processing Announcements'):
        timeFinAnn = str(ann['time'])   # e.g. 2023-07-20T16:54:00.863
        annDate = timeFinAnn.split('T')[0].replace('-', '')
        secCode = ann['sec_code']

        # Fetch outcome of board meeting ann
        new_ann = tracker.getAnnouncements(code=secCode, prevDate=annDate, toDate=annDate, cat='Board Meeting', subCat='Outcome of Board Meeting')
        if new_ann:
            try:
                timeOBM = new_ann['Table'][0]['DT_TM']
                
                # Convert to datetime object for comparison
                dtFinnAnn = datetime.fromisoformat(timeFinAnn)
                dtOBMAnn = datetime.fromisoformat(timeOBM)

                finalTime = dtFinnAnn if dtFinnAnn<dtOBMAnn else dtOBMAnn
                finalType = 'Financial Results' if dtFinnAnn<dtOBMAnn else 'Board Meeting'

                results_df.loc[i, 'Earliest Publish Time'] = finalTime
                results_df.loc[i, 'Earliest Publish Type'] = finalType
            except:
                results_df.loc[i, 'Earliest Publish Time'] = ann['time']
                results_df.loc[i, 'Earliest Publish Type'] = 'Financial Results'
        else:
            results_df.loc[i, 'Earliest Publish Time'] = ann['time']
            results_df.loc[i, 'Earliest Publish Type'] = 'Financial Results'

    results_df.to_csv(os.path.join(base_dir, resFile))
