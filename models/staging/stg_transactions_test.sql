WITH raw_data AS (
    SELECT
        column00 AS row_id,
        CAST(trans_date_trans_time AS TIMESTAMP) AS trans_date_trans_time,
        cc_num,
        merchant,
        category,
        amt,
        first,
        last,
        gender,
        street,
        city,
        state,
        zip,
        lat,
        long,
        city_pop,
        job,
        dob,
        trans_num,
        unix_time,
        merch_lat,
        merch_long,
        is_fraud
    FROM read_csv_auto('{{ var("raw_data_dir") }}/fraudTest.csv')
)
SELECT
    *,
    EXTRACT(HOUR FROM trans_date_trans_time) AS hour,
    (EXTRACT(ISODOW FROM trans_date_trans_time) - 1) AS day_of_week,
    CASE WHEN EXTRACT(HOUR FROM trans_date_trans_time) >= 21 OR EXTRACT(HOUR FROM trans_date_trans_time) < 6 THEN 1 ELSE 0 END AS night
FROM raw_data
