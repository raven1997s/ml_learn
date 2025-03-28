# JSON Data

| Field   | Value               |
|---------|---------------------|
| transaction_id | 1234567890abcdef1234567890abcdef |
| out_trade_no | 20240920123456789 |
| transaction_time | 2024-09-20 15:30:45 |
| transaction_amount | 12345 |
| currency | CNY |
| payment_status | SUCCESS |
| payment_time | 2024-09-20 15:30:46 |
| payment_method | wxpay |
| payer.openid | oxL000000000000000 |
| payer.nickname | 张三 |
| payer.avatar_url | https://example.com/avatar.png |
| payer.country | 中国 |
| payer.province | 广东省 |
| payer.city | 广州市 |
| payer.gender | 1 |
| payee.mchid | 1230000100 |
| payee.sub_mchid | 1230000200 |
| payee.appid | wx1234567890abcdef |
| payee.brand_name | 某某科技有限公司 |
| payee.notify_url | https://example.com/notify |
| goods_detail.0.goods_id | 1001 |
| goods_detail.0.goods_name | iPhone 15 Pro |
| goods_detail.0.quantity | 1 |
| goods_detail.0.price | 899900 |
| goods_detail.0.category | 手机 |
| goods_detail.0.body | A15 芯片，6.1 英寸屏幕 |
| goods_detail.0.goods_url | https://example.com/goods/1001 |
| goods_detail.1.goods_id | 1002 |
| goods_detail.1.goods_name | Apple Watch Series 9 |
| goods_detail.1.quantity | 1 |
| goods_detail.1.price | 399900 |
| goods_detail.1.category | 智能手表 |
| goods_detail.1.body | S9 芯片，41 毫米表盘 |
| goods_detail.1.goods_url | https://example.com/goods/1002 |
| attach | 订单备注信息 |
| fee_detail.0.fee_type | tax |
| fee_detail.0.amount | 1234 |
| fee_detail.0.description | 增值税 |
| fee_detail.1.fee_type | shipping |
| fee_detail.1.amount | 1000 |
| fee_detail.1.description | 快递费 |
| promotion_detail.0.promotion_id | PROMO202409 |
| promotion_detail.0.promotion_name | 开学季满减活动 |
| promotion_detail.0.discount_amount | 10000 |
| promotion_detail.0.discount_type | full_reduction |
| promotion_detail.0.description | 满 1000 减 100 |
| refund_info.refund_status | NOT_REFUND |
| refund_info.refund_id |  |
| refund_info.refund_amount | 0 |
| refund_info.refund_time |  |
| device_info.device_id | 0123456789 |
| device_info.device_type | iOS |
| device_info.device_model | iPhone 15 Pro |
| device_info.os_version | iOS 17.0 |
| device_info.client_ip | 192.168.1.1 |
| risk_info.risk_level | LOW |
| risk_info.risk_score | 0.12 |
| risk_info.risk_detail | 无异常 |
| marketing_info.channel | official_account |
| marketing_info.campaign_id | CAMPAIGN202409 |
| marketing_info.campaign_name | 秋季促销活动 |
| marketing_info.referral_code | REFERRAL123 |
| ext_info.user_cookie | COOKIE123456 |
| ext_info.session_id | SESSION789012 |
| ext_info.custom_param | 自定义参数 |
| receipt_info.receipt_type | electronic |
| receipt_info.receipt_id | RC202409200001 |
| receipt_info.receipt_url | https://example.com/receipt/RC202409200001 |
| operation_records.0.operation_time | 2024-09-20 15:30:45 |
| operation_records.0.operator_id | admin |
| operation_records.0.operation_type | payment |
| operation_records.0.operation_desc | 用户完成支付 |
| operation_records.1.operation_time | 2024-09-20 15:30:46 |
| operation_records.1.operator_id | system |
| operation_records.1.operation_type | notification |
| operation_records.1.operation_desc | 系统发送支付成功通知 |
