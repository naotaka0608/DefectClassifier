
def test_category_manager_load(category_manager):
    """CategoryManagerが正しく設定を読み込めるか"""
    categories = category_manager.get_categories("cause")
    assert "scratches" in categories
    assert "dents" in categories
    assert len(categories) == 2

def test_category_manager_conversion(category_manager):
    """名前とインデックスの変換テスト"""
    # Name -> Index
    idx = category_manager.name_to_index("shape", "circle")
    assert idx == 0  # 順序依存だが、YAMLの読み込み順による。通常はcircleが先なら0
    
    # Index -> Name
    name = category_manager.index_to_name("shape", 0)
    assert name == "circle"

def test_data_manager_load(data_manager):
    """DataManagerがアノテーションを正しく読み込めるか"""
    annotations = data_manager.load_annotations()
    assert len(annotations) == 2
    assert annotations[0]["file_name"] == "test1.jpg"

def test_data_manager_save(data_manager):
    """DataManagerがアノテーションを正しく保存できるか"""
    new_data = [
        {"file_name": "new.jpg", "cause": "c", "shape": "s", "depth": "d"}
    ]
    data_manager.save_annotations(new_data)
    
    loaded = data_manager.load_annotations()
    assert len(loaded) == 1
    assert loaded[0]["file_name"] == "new.jpg"
