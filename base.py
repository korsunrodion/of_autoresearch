import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from db import db, Subscription

RISK_MAP = {
    'No risk': 1,
    'Low': 2,
    'High': 3,
    'Very High': 4,
    'Extreme': 5,
}


def fetch_df(tracking_model_name: str | list[str] | None = None, v1: bool | None = None, selected: bool | None = True) -> pd.DataFrame:
    db.connect(reuse_if_open=True)
    query = Subscription.select()
    if selected is None or selected == True:
        query = query.where(Subscription.selected == True)
    
    if v1:
        query = query.where(Subscription.v1 == True)
        
    if isinstance(tracking_model_name, list):
        query = query.where(Subscription.tracking_model_name.in_(tracking_model_name))
    elif tracking_model_name:
        query = query.where(Subscription.tracking_model_name == tracking_model_name)
    return pd.DataFrame(list(query.dicts()))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df['subscribed_at'] = pd.to_datetime(df['subscribed_at'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['subscribed_at', 'user_name'])
    df['risk_score'] = df['risk_level'].map(RISK_MAP)
    df = df.dropna(subset=['risk_score'])
    df['user_id_num'] = pd.to_numeric(df['user_id'], errors='coerce')
    df['subscribed_ts'] = df['subscribed_at'].astype('int64') // 10**9
    df['total_chargebacks'] = pd.to_numeric(df['total_chargebacks'], errors='coerce').fillna(0)
    return df[['user_name', 'tracking_model_name', 'subscribed_at', 'user_id_num', 'subscribed_ts', 'risk_level', 'risk_score', 'total_chargebacks']].dropna()
