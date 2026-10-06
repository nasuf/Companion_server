from datetime import datetime, timezone
from types import SimpleNamespace as S
import pytest
from app.services.prompting.store import _publication_metadata

NOW=datetime.now(timezone.utc)
def row(**changes):
    return S(**dict(dict(content='default',defaultContent='default',webManaged=False,createdAt=NOW),**changes))
def audit(id='initial',kind='bootstrap',content='default'):
    return S(id=id,changeType=kind,content=content,createdAt=NOW)

@pytest.mark.parametrize('changes,history,expected',[
 ({},[audit()],'default'),
 ({},[],'unverified'),
 ({'webManaged':True},[audit()],'unverified'),
 ({'content':'custom'},[audit()],'unverified'),
 ({},[audit(),audit('unknown','unknown')],'unverified'),
])
def test_no_publication_requires_proven_default(changes,history,expected):
    assert _publication_metadata(row(**changes),history,{})['content_version_type']==expected

def test_missing_row_without_history_is_default():
    assert _publication_metadata(None,[],{})==dict(content_version_type='default',web_version=None,content_version_id=None)

def test_publication_order_uses_immutable_number_not_timestamps_or_revision():
    history=[audit('a','manual_save','older'),audit('b','manual_save','latest'),audit('toggle','disable','latest')]
    result=_publication_metadata(row(content='latest',webManaged=True),history,{'a':8,'b':9})
    assert result==dict(content_version_type='web',web_version=9,content_version_id='b')

def test_content_differing_from_latest_web_publication_is_unverified():
    result=_publication_metadata(row(),[audit(),audit('a','manual_save','custom')],{'a':1})
    assert result==dict(content_version_type='unverified',web_version=None,content_version_id=None)
