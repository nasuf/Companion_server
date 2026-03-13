from neo4j import AsyncGraphDatabase, AsyncDriver

from app.config import settings

_driver: AsyncDriver | None = None


async def get_driver() -> AsyncDriver:
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
    return _driver


async def close_neo4j():
    global _driver
    if _driver:
        await _driver.close()
        _driver = None


async def run_query(query: str, parameters: dict | None = None) -> list[dict]:
    driver = await get_driver()
    async with driver.session() as session:
        result = await session.run(query, parameters or {})
        return [record.data() async for record in result]


async def run_write(query: str, parameters: dict | None = None) -> None:
    driver = await get_driver()
    async with driver.session() as session:
        await session.run(query, parameters or {})


async def neo4j_health() -> bool:
    try:
        driver = await get_driver()
        async with driver.session() as session:
            result = await session.run("RETURN 1 AS ok")
            record = await result.single()
            return record is not None and record["ok"] == 1
    except Exception:
        return False
