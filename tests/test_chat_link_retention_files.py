from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest

pytestmark = pytest.mark.usefixtures("unit_media_scope")

from app.services.chat_links import retention as r, covers
from app.services.chat_links.extraction import LinkMetadata
from app.services.chat_media import storage


def test_unlink_missing_regular_symlink_and_directory(tmp_path,monkeypatch):
    monkeypatch.setattr(storage,'_MEDIA_DIR',tmp_path)
    r._unlink('missing.jpg')
    (tmp_path/'image.jpg').write_bytes(b'safe');r._unlink('image.jpg')
    assert not (tmp_path/'image.jpg').exists()
    (tmp_path/'target').write_bytes(b'keep');(tmp_path/'link.jpg').symlink_to(tmp_path/'target')
    with pytest.raises(ValueError):r._unlink('link.jpg')
    assert (tmp_path/'target').read_bytes()==b'keep'
    (tmp_path/'directory.jpg').mkdir()
    with pytest.raises(ValueError):r._unlink('directory.jpg')
    with pytest.raises(Exception):r._unlink('../outside.jpg')


async def test_cover_capacity_fallback_and_reserved_write_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(storage,'_MEDIA_DIR',tmp_path)
    monkeypatch.setattr(r,'reserve_cover',AsyncMock(return_value=False))
    assert await r.write_cover(user_id='u',conversation_id='c',blob=b'image',mime='image/jpeg') is None
    monkeypatch.setattr(r,'reserve_cover',AsyncMock(return_value=True))
    monkeypatch.setattr(storage,'storage_key_for',lambda *a,**k:'occupied.jpg')
    (tmp_path/'occupied.jpg').write_bytes(b'original')
    with pytest.raises(FileExistsError):await r.write_cover(user_id='u',conversation_id='c',blob=b'new',mime='image/jpeg')
    assert (tmp_path/'occupied.jpg').read_bytes()==b'original'
    metadata=LinkMetadata(source_url='https://example.com',final_url='https://example.com',platform='链接',title='fixture',image_url='https://image.example/a')
    monkeypatch.setattr(covers,'_download_image',AsyncMock(return_value=(b'image','image/jpeg')))
    monkeypatch.setattr(covers,'write_cover',AsyncMock(return_value=None))
    result=await covers.cache_link_cover(user_id='u',conversation_id='c',metadata=metadata)
    assert result.metadata.image_url==metadata.image_url and result.extra_metadata['cover_cache_status']=='capacity_limited'
