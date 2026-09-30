"""Standard-library HTTP contract test; no legacy pytest dependencies."""
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from workbench.__main__ import LabHandler


class QuietHandler(LabHandler):
    def log_message(self, *_args):
        pass


class LauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.server = ThreadingHTTPServer(
            ('127.0.0.1', 0), partial(QuietHandler, directory=str(root))
        )
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_entry_page_and_module_mime(self):
        with urlopen(self.url, timeout=5) as response:
            self.assertIn(b'A laboratory for motion', response.read())
            self.assertEqual(response.headers['Cache-Control'], 'no-store')
        with urlopen(self.url + '/worker.mjs', timeout=5) as response:
            self.assertEqual(response.headers.get_content_type(), 'text/javascript')
            self.assertEqual(response.headers['X-Content-Type-Options'], 'nosniff')
            self.assertIn(b'./physics.mjs', response.read())

    def test_repository_files_are_outside_served_directory(self):
        for path in ['/../.git/config', '/%2e%2e/.git/config', '/README.md']:
            with self.assertRaises(HTTPError) as error:
                urlopen(self.url + path, timeout=5)
            self.assertEqual(error.exception.code, 404)


if __name__ == '__main__':
    unittest.main()
