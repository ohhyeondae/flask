
from flask import Flask, request, jsonify
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# SARIMA 예측 함수
def sarima_forecast(data, current_date):
    # 날짜 변환
    data['거래일시'] = pd.to_datetime(data['거래일시'], errors='coerce')

    # 이전 연도의 동일 날짜 기준
    previous_year = current_date.year - 1
    current_date_previous_year = current_date.replace(year=previous_year)
    start_date = current_date_previous_year - timedelta(days=60)
    end_date = current_date_previous_year

    # 60일 데이터 필터링
    filtered_data = data[(data['거래일시'] >= start_date) & (data['거래일시'] <= end_date)]

    # 예측 결과를 저장할 데이터프레임
    forecasted_sales = pd.DataFrame()

    # 각 품목별로 SARIMA 모델 실행
    for product in filtered_data['상품'].dropna().unique():
        # 일별 판매량 집계
        product_data = filtered_data[filtered_data['상품'] == product].resample('D', on='거래일시').size()

        if len(product_data) > 12:  # 최소 12개의 데이터 필요
            try:
                model = SARIMAX(product_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
                model_fit = model.fit(disp=False)

                # 7일 예측
                forecast = model_fit.get_forecast(steps=7)
                forecast_index = pd.date_range(start=current_date + timedelta(days=1), periods=7, freq='D')
                predicted_values = np.ceil(forecast.predicted_mean.values).astype(int)
                predicted_values = np.maximum(predicted_values, 0)  # 음수는 0으로 처리

                # 예측 결과 저장
                forecasted_sales[product] = pd.Series(data=predicted_values, index=forecast_index)

            except Exception as e:
                print(f"품목 '{product}' 예측 중 오류 발생: {e}")

    forecasted_sales.reset_index(inplace=True)
    forecasted_sales.rename(columns={"index": "날짜"}, inplace=True)
    return forecasted_sales

# 파일 업로드 및 예측 처리 API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 클라이언트로부터 파일 및 현재 날짜 수신
        file = request.files['file']
        current_date = datetime.strptime(request.form['current_date'], '%Y-%m-%d')
        
        # CSV 파일 읽기
        data = pd.read_csv(file)

        # SARIMA 예측 수행
        forecasted_sales = sarima_forecast(data, current_date)

        # JSON으로 결과 반환
        response = forecasted_sales.to_dict(orient='records')
        return jsonify({'status': 'success', 'forecast': response})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# 예측 결과 저장 API
@app.route('/save_forecast', methods=['POST'])
def save_forecast():
    try:
        # 클라이언트로부터 파일 및 현재 날짜 수신
        file = request.files['file']
        current_date = datetime.strptime(request.form['current_date'], '%Y-%m-%d')

        # CSV 파일 읽기
        data = pd.read_csv(file)

        # SARIMA 예측 수행
        forecasted_sales = sarima_forecast(data, current_date)

        # 결과를 CSV로 저장
        output_file_path = os.path.join(os.getcwd(), 'forecasted_sales_next_7_days.csv')
        forecasted_sales.to_csv(output_file_path, encoding='EUC-KR', index=False)

        return jsonify({'status': 'success', 'message': f'File saved at {output_file_path}'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
