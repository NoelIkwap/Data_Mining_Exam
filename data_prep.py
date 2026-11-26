import pandas as pd,numpy as np
U='/mnt/user-data/uploads/'
o=pd.read_csv(U+'olist_orders_dataset.csv',parse_dates=['order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date'])
it=pd.read_csv(U+'olist_order_items_dataset.csv',parse_dates=['shipping_limit_date'])
p=pd.read_csv(U+'olist_products_dataset.csv').rename(columns={'product_name_lenght':'name_len','product_description_lenght':'desc_len'})
tr=pd.read_csv(U+'product_category_name_translation.csv')
c=pd.read_csv(U+'olist_customers_dataset.csv'); s=pd.read_csv(U+'olist_sellers_dataset.csv')
g=pd.read_csv(U+'olist_geolocation_dataset.csv'); pay=pd.read_csv(U+'olist_order_payments_dataset.csv'); r=pd.read_csv(U+'olist_order_reviews_dataset.csv',parse_dates=['review_creation_date'])
# geo: one point per zip (median), drop exact dups
geo=g.groupby('geolocation_zip_code_prefix')[['geolocation_lat','geolocation_lng']].median()
# category
tr=pd.concat([tr,pd.DataFrame({'product_category_name':['pc_gamer','portateis_cozinha_e_preparadores_de_alimentos'],'product_category_name_english':['pc_gamer','portable_kitchen_food_preparers']})])
p=p.merge(tr,how='left',on='product_category_name'); p['category']=p.product_category_name_english.fillna('unknown')
p['volume_cm3']=p.product_length_cm*p.product_height_cm*p.product_width_cm
for col in ['product_weight_g','volume_cm3']:
    p.loc[p[col]==0,col]=np.nan
# reviews: one per order (latest)
r1=r.sort_values('review_creation_date').drop_duplicates('order_id',keep='last')[['order_id','review_score']]
# item level
I=it.merge(p[['product_id','category','product_weight_g','volume_cm3','product_photos_qty','desc_len','name_len']],on='product_id',how='left') \
    .merge(s,on='seller_id',how='left').merge(o,on='order_id',how='left').merge(c,on='customer_id',how='left')
I=I.join(geo.rename(columns={'geolocation_lat':'s_lat','geolocation_lng':'s_lng'}),on='seller_zip_code_prefix') \
    .join(geo.rename(columns={'geolocation_lat':'c_lat','geolocation_lng':'c_lng'}),on='customer_zip_code_prefix')
def hav(a,b,c_,d):
    a,b,c_,d=map(np.radians,[a,b,c_,d]); x=np.sin((c_-a)/2)**2+np.cos(a)*np.cos(c_)*np.sin((d-b)/2)**2
    return 6371*2*np.arcsin(np.sqrt(x))
I['dist_km']=hav(I.s_lat,I.s_lng,I.c_lat,I.c_lng)
I.to_pickle('items.pkl')
# order level
pp=pay.groupby('order_id').agg(pay_value=('payment_value','sum'),n_pay=('payment_sequential','count'),installments=('payment_installments','max'),
   pay_type=('payment_type',lambda x:x.mode().iat[0]))
O=I.groupby('order_id').agg(n_items=('order_item_id','count'),n_sellers=('seller_id','nunique'),n_cats=('category','nunique'),price=('price','sum'),freight=('freight_value','sum'),
   weight=('product_weight_g','sum'),volume=('volume_cm3','sum'),dist_km=('dist_km','mean'),max_price=('price','max'),seller_state=('seller_state','first'),
   category=('category','first')).reset_index()
O=o.merge(O,on='order_id',how='left').merge(c,on='customer_id').merge(pp,on='order_id',how='left').merge(r1,on='order_id',how='left')
O['days_total']=(O.order_delivered_customer_date-O.order_purchase_timestamp).dt.total_seconds()/86400
O['est_days']=(O.order_estimated_delivery_date-O.order_purchase_timestamp).dt.total_seconds()/86400
O['late']=(O.order_delivered_customer_date>O.order_estimated_delivery_date).astype(float).where(O.order_delivered_customer_date.notna())
O.to_pickle('orders.pkl')
print(O.shape, O.late.mean())
print('dist NaN',I.dist_km.isna().mean().round(4))
print('late by purchase month'); d=O.dropna(subset=['late']); print(d.groupby(d.order_purchase_timestamp.dt.to_period('Q')).agg(n=('late','size'),late=('late','mean'),days=('days_total','mean')).round(3).to_string())
print(d.groupby('customer_state').agg(n=('late','size'),late=('late','mean'),days=('days_total','mean')).sort_values('late',ascending=False).head(6).round(3).to_string())
print('review by late',d.groupby('late').review_score.mean().round(2).to_dict())
print(d[['dist_km','freight','weight','days_total']].corr(method='spearman').round(2).to_string())
