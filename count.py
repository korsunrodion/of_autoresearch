from db import db, Subscription

db.connect()
print(Subscription.select().count())
db.close()
