from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from plugin.server.local_app_bridge.contracts import LaunchMaterial

MAX_LAUNCH_MATERIAL_BYTES = 4096


async def write_launch_material(
    writer: asyncio.StreamWriter,
    material: LaunchMaterial,
) -> None:
    """Write the sole secret through a pipe/stdin, never argv or logs."""

    encoded = (
        json.dumps(material.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_LAUNCH_MATERIAL_BYTES:
        raise ValueError("launch material exceeds the bounded pipe contract")
    writer.write(encoded)
    await writer.drain()
    writer.close()
    await writer.wait_closed()


async def launch_local_app(
    executable: str | Path,
    *,
    args: Sequence[str] = (),
    material: LaunchMaterial,
) -> asyncio.subprocess.Process:
    """Launch an app with public argv and deliver credentials only over stdin."""

    if any(material.launch_code in str(argument) for argument in args):
        raise ValueError("launch code must not appear in process arguments")
    process = await asyncio.create_subprocess_exec(
        str(executable),
        *(str(argument) for argument in args),
        stdin=asyncio.subprocess.PIPE,
    )
    if process.stdin is None:
        process.kill()
        await process.wait()
        raise RuntimeError("local app stdin pipe is unavailable")
    try:
        await write_launch_material(process.stdin, material)
    except BaseException:
        process.kill()
        await process.wait()
        raise
    return process
