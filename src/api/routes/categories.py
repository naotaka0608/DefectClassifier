"""カテゴリ管理エンドポイント"""

from fastapi import APIRouter, HTTPException

from src.api.schemas.response import CategoriesResponse
from src.core.category_manager import CategoryManager
from src.core.config import DEFAULT_CATEGORIES_CONFIG

router = APIRouter(prefix="/api/v1", tags=["categories"])

# カテゴリマネージャー（実際の実装では依存性注入を使用）
_category_manager = CategoryManager(DEFAULT_CATEGORIES_CONFIG)


@router.get("/categories", response_model=CategoriesResponse)
async def get_categories() -> CategoriesResponse:
    """カテゴリ一覧を取得"""
    return CategoriesResponse(
        cause_categories=_category_manager.get_categories("cause"),
        shape_categories=_category_manager.get_categories("shape"),
        depth_categories=_category_manager.get_categories("depth"),
    )


@router.post("/categories/reload")
async def reload_categories() -> dict:
    """カテゴリを再読み込み"""
    try:
        _category_manager.reload()
        return {"success": True, "message": "Categories reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
