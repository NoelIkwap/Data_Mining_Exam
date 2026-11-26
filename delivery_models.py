import pandas as pd,numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HC, HistGradientBoostingRegressor as HR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score,average_precision_score,mean_absolute_error,precision_score,recall_score,r2_score
O=pd.read_pickle('orders.pkl')
I=pd.read_pickle('items.pkl'); O=O.merge(I.drop_duplicates('order_id')[['order_id','seller_id']],on='order_id',how='left')
D=O[(O.order_status=='delivered')&O.order_delivered_customer_date.notna()&O.dist_km.notna()].copy()
D=D[(D.days_total<=90)]  # drop extreme outliers (top ~0.2%) for regression stability
D['approve_hrs']=(D.order_approved_at-D.order_purchase_timestamp).dt.total_seconds()/3600
D['approve_hrs']=D.approve_hrs.clip(0,200)
t=D.order_purchase_timestamp
D['month']=t.dt.month; D['dow']=t.dt.dayofweek; D['hour']=t.dt.hour
D['same_state']=(D.customer_state==D.seller_state).astype(int)
D['freight_ratio']=D.freight/D.price.clip(lower=1)
for c in ['customer_state','seller_state','category','pay_type']: D[c]=D[c].astype('category')
D['week']=t.dt.to_period('W').astype(str)
D['week_orders']=D.groupby('week').order_id.transform('count')
D['est_per_km']=D.est_days/(D.dist_km+50)
num=['price','freight','freight_ratio','weight','volume','n_items','n_sellers','dist_km','est_days','approve_hrs','month','dow','hour','same_state','installments']
cat=['customer_state','seller_state','category','pay_type']
feats=num+cat
cut=pd.Timestamp('2018-04-01')
trm=D[t<cut]; gm=trm.late.mean()
# seller history from TRAIN period only; leave-one-out for train rows so a row never sees its own outcome
istr=(D.order_purchase_timestamp<cut)
g=D[istr].groupby('seller_id'); K=20
tl,tc,td=g.late.sum(),g.late.count(),g.days_total.sum(); gd=D[istr].days_total.mean()
def enc(col_sum,own,gmean):
    sm=D.seller_id.map(col_sum).fillna(0); cn=D.seller_id.map(tc).fillna(0)
    sm=sm-own.where(istr,0); cn=cn-istr.astype(int)
    return (sm+gmean*K)/(cn+K)
D['sell_late']=enc(tl,D.late,gm); D['sell_days']=enc(td,D.days_total,gd)
pass
cut=pd.Timestamp('2018-04-01'); tr=D[t<cut]; te=D[t>=cut]
print('train',len(tr),'test',len(te),'late rate train/test',tr.late.mean().round(3),te.late.mean().round(3))
# classification
clf=HC(max_iter=300,learning_rate=0.05,max_leaf_nodes=20,categorical_features='from_dtype',class_weight='balanced',random_state=0).fit(tr[feats],tr.late)
pr=clf.predict_proba(te[feats])[:,1]
print('AUC',round(roc_auc_score(te.late,pr),3),'PR-AUC',round(average_precision_score(te.late,pr),3),'baseline PR',te.late.mean().round(3))
# baseline: est_days only / distance
for n in ['est_days','dist_km']:
    print(' single-feature',n,round(roc_auc_score(te.late,-te[n] if n=='est_days' else te[n]),3))
for th in [0.5,0.6,0.7,0.8]:
    pl=pr>=th; print(th,'flagged %',pl.mean().round(3),'precision',round(precision_score(te.late,pl),3),'recall',round(recall_score(te.late,pl),3))
# regression
reg=HR(max_iter=400,learning_rate=0.05,max_leaf_nodes=20,categorical_features='from_dtype',loss='absolute_error',random_state=0).fit(tr[feats],tr.days_total)
pd_=reg.predict(te[feats])
print('MAE model',round(mean_absolute_error(te.days_total,pd_),2),'R2',round(r2_score(te.days_total,pd_),3))
print('MAE baseline = Olist estimate',round(mean_absolute_error(te.days_total,te.est_days),2),'| baseline median',round(mean_absolute_error(te.days_total,np.full(len(te),tr.days_total.median())),2))
ln=LinearRegression().fit(tr[['dist_km','freight','weight','est_days']].fillna(0),tr.days_total); print('linear MAE',round(mean_absolute_error(te.days_total,ln.predict(te[['dist_km','freight','weight','est_days']].fillna(0))),2))
print('within 3 days',round((abs(te.days_total-pd_)<=3).mean(),3),'est within 3',round((abs(te.days_total-te.est_days)<=3).mean(),3))
# late via regression: predicted days > est days
pl=(pd_>te.est_days-3); print('late via reg (pred>est-3d) prec',round(precision_score(te.late,pl),3),'rec',round(recall_score(te.late,pl),3),'flag',pl.mean().round(3))
# permutation importance (sample)
from sklearn.inspection import permutation_importance
s=te.sample(8000,random_state=0)
for name,m,y,sc in [('clf',clf,s.late,'roc_auc'),('reg',reg,s.days_total,'neg_mean_absolute_error')]:
    pi=permutation_importance(m,s[feats],y,scoring=sc,n_repeats=3,random_state=0,n_jobs=-1)
    print(name,pd.Series(pi.importances_mean,feats).sort_values(ascending=False).head(7).round(3).to_dict())
# quarter breakdown
te=te.assign(p=pr); print(te.groupby(te.order_purchase_timestamp.dt.to_period('M')).apply(lambda x:pd.Series(dict(n=len(x),late=x.late.mean(),auc=roc_auc_score(x.late,x.p) if x.late.nunique()>1 else np.nan))).round(3).to_string())
