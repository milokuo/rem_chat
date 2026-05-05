"""Phase 3: Verify POST / returns full_prompt, raw_response, model_name.

Run in clip_env from predictors/clip_iu/:
    python -m pytest test_chat_engine_trace_fields.py -v
"""
import json
import sys
import unittest
from unittest.mock import MagicMock, patch

# Use config defaults — no extra CLI args
sys.argv = ['chat_engine.py']

# deep_translator is not installed in clip_env; stub it before chat_engine imports it
_dt_stub = MagicMock()
_dt_stub.GoogleTranslator = MagicMock()
sys.modules.setdefault('deep_translator', _dt_stub)


def _fake_gpt_response(text: str) -> MagicMock:
    """Minimal OpenAI response stub."""
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _fake_stream_chunk(text: str) -> MagicMock:
    """Minimal streamed OpenAI chunk stub."""
    delta = MagicMock()
    delta.content = text
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


def _fake_stream_response(text: str):
    midpoint = max(1, len(text) // 2)
    return [
        _fake_stream_chunk(text[:midpoint]),
        _fake_stream_chunk(text[midpoint:]),
    ]


RAW_REPLY = '9. Sure!\nJSON_OUTPUT: {"reply": "Sure!", "strategy": "[Others]"}'
STREAM_REPLY = 'Sure thing!'


class TestChatEngineTraceFields(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Patch openai.OpenAI before importing so no real client is created
        cls._patcher = patch('openai.OpenAI')
        cls.mock_openai_cls = cls._patcher.start()
        cls.mock_client = MagicMock()
        cls.mock_openai_cls.return_value = cls.mock_client

        import chat_engine as ce
        cls.ce = ce

        # Recreate globals that __main__ normally sets
        cls.ce._socialREMChat = cls.ce.SocialREMChat(lang='en')
        cls.ce._socialREMChat.context = [{'Assistant': 'Hello!'}]
        cls.ce.end_trigger = 'conversation over'

        cls.flask_client = cls.ce.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls._patcher.stop()

    def setUp(self):
        """Reset shared chat state before each test."""
        self.ce._socialREMChat.context = [{'Assistant': 'Hello!'}]
        self.ce._socialREMChat.caption_str = 'a birthday party'
        self.ce._socialREMChat.obj_str = 'cake, candles'
        self.ce._socialREMChat.retrieved_context = ''
        self.mock_client.chat.completions.create.return_value = _fake_gpt_response(RAW_REPLY)

    def _post(self, payload: dict) -> dict:
        resp = self.flask_client.post(
            '/',
            data=json.dumps(payload),
            content_type='application/json',
        )
        return json.loads(resp.data)

    def _post_stream(self, payload: dict) -> list:
        resp = self.flask_client.post(
            '/stream',
            data=json.dumps(payload),
            content_type='application/json',
        )
        events = []
        for block in resp.data.decode('utf-8').strip().split('\n\n'):
            if block.startswith('data: '):
                events.append(json.loads(block[6:]))
        return events

    # ── full_prompt ──────────────────────────────────────────────────────

    def test_normal_chat_has_full_prompt(self):
        body = self._post({'user_message': 'Hi there'})
        self.assertIn('full_prompt', body, 'full_prompt missing from normal chat response')

    def test_full_prompt_is_list(self):
        body = self._post({'user_message': 'Hi there'})
        self.assertIsInstance(body['full_prompt'], list)
        self.assertGreater(len(body['full_prompt']), 0)

    def test_full_prompt_messages_have_role_and_content(self):
        body = self._post({'user_message': 'Hi there'})
        for msg in body['full_prompt']:
            self.assertIn('role', msg)
            self.assertIn('content', msg)

    # ── raw_response ─────────────────────────────────────────────────────

    def test_normal_chat_has_raw_response(self):
        body = self._post({'user_message': 'Hi there'})
        self.assertIn('raw_response', body, 'raw_response missing from normal chat response')

    def test_raw_response_is_string(self):
        body = self._post({'user_message': 'Hi there'})
        self.assertIsInstance(body['raw_response'], str)

    def test_raw_response_equals_gpt_output(self):
        """raw_response must be the unprocessed GPT text, not the parsed reply."""
        body = self._post({'user_message': 'Hi there'})
        self.assertEqual(body['raw_response'], RAW_REPLY)

    # ── model_name ───────────────────────────────────────────────────────

    def test_normal_chat_has_model_name(self):
        body = self._post({'user_message': 'Hi there'})
        self.assertIn('model_name', body, 'model_name missing from normal chat response')

    def test_model_name_is_non_empty_string(self):
        body = self._post({'user_message': 'Hi there'})
        self.assertIsInstance(body['model_name'], str)
        self.assertGreater(len(body['model_name']), 0)

    # ── reset (new image) also returns trace fields ──────────────────────

    def test_reset_has_full_prompt(self):
        body = self._post({'reset': True, 'caption_str': 'a park', 'obj_str': 'tree'})
        self.assertIn('full_prompt', body)

    def test_reset_has_raw_response(self):
        body = self._post({'reset': True, 'caption_str': 'a park', 'obj_str': 'tree'})
        self.assertIn('raw_response', body)

    def test_reset_has_model_name(self):
        body = self._post({'reset': True, 'caption_str': 'a park', 'obj_str': 'tree'})
        self.assertIn('model_name', body)

    # ── existing fields must not break ───────────────────────────────────

    def test_existing_fields_still_present(self):
        body = self._post({'user_message': 'Hello'})
        self.assertIn('return_message', body)
        self.assertIn('last', body)
        self.assertIn('timing', body)

    def test_return_message_matches_parsed_reply(self):
        """return_message must be the parsed/clean reply, not the raw GPT output."""
        body = self._post({'user_message': 'Hello'})
        self.assertEqual(body['return_message'], 'Sure!')

    # ---- streaming path intentionally uses the fast direct-reply prompt ----

    def test_stream_prompt_uses_direct_reply_prompt(self):
        """Unchecked precise-mode streaming should skip CoT/JSON so tokens can stream."""
        self.mock_client.chat.completions.create.return_value = _fake_stream_response(STREAM_REPLY)
        events = self._post_stream({'user_message': 'Hi there'})
        done = events[-1]
        prompt = done['full_prompt'][0]['content']

        self.assertIn('Do not output labels, bullets, analysis, structured templates, or JSON', prompt)
        self.assertIn('strategy explanations', prompt)
        self.assertIn('Never mention the strategy name', prompt)
        self.assertIn("same language as the user's latest utterance", prompt)
        self.assertNotIn('Then select a Support Strategy', prompt)
        self.assertNotIn('Support Strategies Definitions', prompt)
        self.assertNotIn('explains why', prompt)
        self.assertNotIn('Demand Format', prompt)
        self.assertNotIn('JSON_OUTPUT', prompt)

    def test_stream_forwards_token_chunks_and_keeps_raw_response(self):
        """Browser receives model chunks as they arrive; dashboard keeps raw text."""
        self.mock_client.chat.completions.create.return_value = _fake_stream_response(STREAM_REPLY)
        events = self._post_stream({'user_message': 'Hi there'})
        token_events = [event for event in events if 'token' in event]
        token_text = ''.join(event.get('token', '') for event in token_events)
        done = events[-1]

        self.assertGreater(len(token_events), 1)
        self.assertEqual(token_text, STREAM_REPLY)
        self.assertEqual(done['full'], STREAM_REPLY)
        self.assertEqual(done['raw_response'], STREAM_REPLY)
        self.assertIn('model_name', done)
        self.assertIn('gpt_ms', done['timing'])


if __name__ == '__main__':
    unittest.main()
