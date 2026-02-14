
def test_health_check(test_client):
    """ヘルスチェックのテスト"""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint(test_client, mock_predictor):
    """推論エンドポイントのテスト"""
    # Base64ダミー画像
    dummy_image = "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
    
    payload = {
        "image_base64": dummy_image,
        "return_confidence": True
    }
    
    response = test_client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["success"] is True
    assert data["cause"]["label"] == "scratches"
    
    # Mockが呼ばれたか確認
    mock_predictor.predict_from_base64.assert_called_once()

def test_predict_batch_endpoint(test_client, mock_predictor):
    """バッチ推論エンドポイントのテスト"""
    dummy_image = "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
    
    payload = {
        "images": [dummy_image, dummy_image],
        "return_confidence": True
    }
    
    response = test_client.post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["success"] is True
    assert len(data["results"]) == 2
