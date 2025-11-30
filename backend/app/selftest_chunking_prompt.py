from app.main import fixed_chunk_text, paragraph_chunk_text, build_rag_answer_prompt


def _test_fixed_chunk_text():
    text = "あ" * 2500
    chunks = fixed_chunk_text(text, chunk_size=1000, chunk_overlap=0)
    assert len(chunks) == 3
    assert "".join(chunks) == text


def _test_paragraph_chunk_text():
    text = "第一段落" "\n\n" "第二段落" "\n\n" "第三段落"
    chunks = paragraph_chunk_text(text)
    assert chunks == ["第一段落", "第二段落", "第三段落"]


def _test_build_rag_answer_prompt():
    context = "これはコンテキストです。"
    question = "これは質問ですか？"
    prompt = build_rag_answer_prompt(context, question)
    assert "コンテキスト" in prompt
    assert context in prompt
    assert "質問" in prompt
    assert question in prompt
    assert "回答" in prompt


def run_all_tests():
    _test_fixed_chunk_text()
    _test_paragraph_chunk_text()
    _test_build_rag_answer_prompt()


if __name__ == "__main__":
    run_all_tests()
    print("selftest_chunking_prompt: OK")
