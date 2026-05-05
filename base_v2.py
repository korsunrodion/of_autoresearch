import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from db import db, TrackingLinkSubscriber

RISK_TO_DB = {
    'No risk':   'no risk',
    'Low':       'low',
    'High':      'high',
    'Very High': 'very high',
    'Extreme':   'extreme',
}


def fetch_df() -> pd.DataFrame:
    """Fetch ALL rows (warm context + new cold rows), no is_processed filter."""
    db.connect(reuse_if_open=True)
    query = TrackingLinkSubscriber.select()
    return pd.DataFrame(list(query.dicts()))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        'username':          'user_name',
        'tracking_link_id':  'tracking_model_name',
        'subscription_date': 'subscribed_at',
        'user_id':           'user_id_num',
    })

    df['subscribed_at'] = pd.to_datetime(df['subscribed_at'], errors='coerce')
    df = df.dropna(subset=['subscribed_at', 'user_name'])

    df['risk_level'] = (
        df['risk_level'].fillna('no risk')
        .str.title()
        .replace({'No Risk': 'No risk'})
    )
    df['risk_score'] = df['risk_level'].map(
        {'No risk': 1, 'Low': 2, 'High': 3, 'Very High': 4, 'Extreme': 5}
    )
    df = df.dropna(subset=['risk_score'])

    df['subscribed_ts']     = df['subscribed_at'].astype('int64') // 10 ** 9
    df['total_chargebacks'] = pd.to_numeric(
        df.get('total_chargebacks', 0), errors='coerce'
    ).fillna(0)

    cols = [
        'user_name', 'tracking_model_name', 'subscribed_at',
        'user_id_num', 'subscribed_ts', 'risk_level', 'risk_score',
        'total_chargebacks',
    ]
    if 'is_internal_data' in df.columns:
        cols.append('is_internal_data')

    return df[cols].dropna(subset=['user_id_num'])


def update_risk_levels(predictions: dict[str, str]) -> int:
    """
    Write predicted risk levels back to DB, skipping warm (is_internal_data=True) rows
    so ground-truth labels are not overwritten.
    """
    db.connect(reuse_if_open=True)

    updated = 0
    items = list(predictions.items())
    chunk_size = 500
    for i in range(0, len(items), chunk_size):
        chunk = items[i:i + chunk_size]
        for username, risk_title in chunk:
            risk_db = RISK_TO_DB.get(risk_title)
            if risk_db is None:
                continue
            query = (
                TrackingLinkSubscriber
                .update(risk_level=risk_db)
                .where(TrackingLinkSubscriber.username == username)
            )
            # Only touch cold rows if the column exists
            try:
                query = query.where(
                    TrackingLinkSubscriber.is_internal_data == False
                )
            except Exception:
                pass
            updated += query.execute()

    return updated
