import os
from dotenv import load_dotenv
from peewee import BooleanField, PostgresqlDatabase, Model, CharField, DateTimeField, AutoField, SQL
from datetime import datetime

from urllib.parse import urlparse

load_dotenv()

def _make_db():
  url = os.environ.get('DATABASE_URL')
  if url:
    p = urlparse(url)
    return PostgresqlDatabase(
      p.path.lstrip('/'),
      user=p.username,
      password=p.password or '',
      host=p.hostname,
      port=p.port or 5432,
    )
  return PostgresqlDatabase(
    os.environ.get('DB_NAME', 'of_parser'),
    user=os.environ.get('DB_USER', 'postgres'),
    password=os.environ.get('DB_PASSWORD', ''),
    host=os.environ.get('DB_HOST', 'localhost'),
    port=int(os.environ.get('DB_PORT', 5432)),
  )

db = _make_db()

class Subscription(Model):
  id = AutoField()
  user_id = CharField(null=True)
  user_name = CharField(null=True)
  subscribed_at = CharField(null=True)
  timeline = CharField(null=True)
  total_chargebacks = CharField(null=True)
  total = CharField(null=True)
  interrupted = CharField(null=True)
  risk_level = CharField(null=True)
  v1 = BooleanField(null=True)
  selected = BooleanField(null=True)
  
  model_id = CharField(null=True)
  tracking_model_id = CharField(null=True)
  tracking_model_name = CharField(null=True)
  
  updated_at = DateTimeField(default=datetime.now)

  class Meta:
    database = db
    table_name = 'subscriptions'
    indexes = (
      (('tracking_model_name', 'user_id'), True),  # True = unique
    )

def init_db():
  db.connect()
  db.create_tables([Subscription])

def upsert_row(data: dict):
  row = {
    'user_id': data.get('userId'),
    'user_name': data.get('userName'),
    'subscribed_at': data.get('subscribedAt'),
    'timeline': data.get('timeline'),
    'total_chargebacks': data.get('totalChargebacks'),
    'total': data.get('total'),
    'interrupted': data.get('interrupted'),
    'risk_level': data.get('riskLevel'),
    'model_id': data.get('modelId'),
    'tracking_model_id': data.get('trackingModelId'),
    'tracking_model_name': data.get('trackingModelName'),
    'updated_at': datetime.now(),
    'v1': data.get('v1'),
    'selected': data.get('selected'),
  }
  cursor = Subscription.insert(row).on_conflict(  # pylint: disable=no-value-for-parameter
    conflict_target=[Subscription.tracking_model_name, Subscription.user_id],
    update=row,
  ).returning(SQL('xmax')).tuples().execute()
  xmax = next(iter(cursor))[0]
  return 'created' if xmax == 0 else 'updated'
