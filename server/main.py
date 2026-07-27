import asyncio
import logging
import os
from aiohttp import web
import protocol

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

TCP_PORT  = int(os.getenv('JIMI_PORT', '47755'))
HTTP_PORT = int(os.getenv('API_PORT',  '8090'))

_devices: dict[str, 'Device'] = {}


class Device:
    def __init__(self, writer: asyncio.StreamWriter, addr: str):
        self.writer = writer
        self.addr   = addr
        self.imei: str | None = None
        self._sn  = 1
        self._buf = b''

    async def _send(self, data: bytes):
        self.writer.write(data)
        await self.writer.drain()

    async def send_command(self, cmd: str):
        log.info(f'[{self.imei}] → {cmd!r}')
        await self._send(protocol.command_packet(cmd, self._sn))
        self._sn += 1

    async def run(self, reader: asyncio.StreamReader):
        log.info(f'[{self.addr}] connected')
        try:
            while True:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                self._buf += chunk
                packets, self._buf = protocol.parse(self._buf)
                for proto, content, sn in packets:
                    await self._dispatch(proto, content, sn)
        except (ConnectionResetError, asyncio.IncompleteReadError):
            pass
        finally:
            if self.imei and _devices.get(self.imei) is self:
                del _devices[self.imei]
            log.info(f'[{self.imei or self.addr}] disconnected')

    async def _dispatch(self, proto: int, content: bytes, sn: int):
        if proto == protocol.LOGIN:
            self.imei = protocol.decode_imei(content)
            _devices[self.imei] = self
            log.info(f'[{self.addr}] login IMEI={self.imei}')
            await self._send(protocol.login_ack(sn))

        elif proto == protocol.HEARTBEAT:
            log.debug(f'[{self.imei}] heartbeat')
            await self._send(protocol.heartbeat_ack(sn))

        elif proto == protocol.TIME_CAL:
            log.debug(f'[{self.imei}] time calibration request')
            await self._send(protocol.time_ack(sn))

        elif proto == protocol.CMD_RESP:
            # content: server_flags(4) + encoding(1) + message
            if len(content) >= 5:
                enc = content[4]
                msg = content[5:].decode('ascii' if enc == 1 else 'utf-16-be', errors='replace')
                log.info(f'[{self.imei}] ← {msg!r}')

        elif proto == protocol.GPS:
            log.debug(f'[{self.imei}] GPS ({len(content)} bytes)')

        else:
            log.debug(f'[{self.imei}] proto=0x{proto:02X} ({len(content)} bytes)')


async def _tcp_handler(reader, writer):
    dev = Device(writer, ':'.join(str(x) for x in writer.get_extra_info('peername')))
    await dev.run(reader)
    writer.close()


# ── REST API ──────────────────────────────────────────────────────────────────

async def get_devices(request):
    return web.json_response({imei: d.addr for imei, d in _devices.items()})


async def post_command(request):
    imei = request.match_info['imei']
    body = await request.json()
    cmd  = body.get('command', '').strip()
    if not cmd:
        return web.json_response({'error': 'command required'}, status=400)
    dev = _devices.get(imei)
    if not dev:
        return web.json_response({'error': 'device not connected'}, status=404)
    await dev.send_command(cmd)
    return web.json_response({'sent': cmd})


async def post_stream_start(request):
    imei = request.match_info['imei']
    dev  = _devices.get(imei)
    if not dev:
        return web.json_response({'error': 'device not connected'}, status=404)
    await dev.send_command('RTMP,ON,INOUT')
    return web.json_response({'streaming': True, 'imei': imei})


async def post_stream_stop(request):
    imei = request.match_info['imei']
    dev  = _devices.get(imei)
    if not dev:
        return web.json_response({'error': 'device not connected'}, status=404)
    await dev.send_command('RTMP,OFF')
    return web.json_response({'streaming': False, 'imei': imei})


async def main():
    tcp = await asyncio.start_server(_tcp_handler, '0.0.0.0', TCP_PORT)
    log.info(f'Jimi TCP server on :{TCP_PORT}')

    app = web.Application()
    app.router.add_get( '/devices',              get_devices)
    app.router.add_post('/command/{imei}',        post_command)
    app.router.add_post('/stream/start/{imei}',   post_stream_start)
    app.router.add_post('/stream/stop/{imei}',    post_stream_stop)

    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, '0.0.0.0', HTTP_PORT).start()
    log.info(f'HTTP API on :{HTTP_PORT}')

    async with tcp:
        await tcp.serve_forever()


if __name__ == '__main__':
    asyncio.run(main())
