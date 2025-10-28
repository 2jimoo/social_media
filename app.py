# Streamlit >= 1.25 권장 (st.chat_input / st.chat_message 사용)
# pip install streamlit requests beautifulsoup4 langchain langchain-community sentence-transformers faiss-cpu

import json
import time
from typing import Dict, Iterator, List, Optional

import requests
import streamlit as st

# ----- LangChain / RAG 관련 -----
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="Streamlit Chat + Sidebar (Ollama + RAG)", layout="centered")

# ---------------------- 설정 / 세션 상태 ----------------------
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []
if "mode" not in st.session_state:
    st.session_state.mode = "Facebook"
if "system_prompts" not in st.session_state:
    st.session_state.system_prompts = {
        "Facebook": "You are a Social Media Policy Consultant specializing in Facebook. Answer in Korean. Be concise, cite relevant Facebook policies when helpful (paraphrase).",
        "Instagram": "You are a Social Media Policy Consultant specializing in Instagram. Answer in Korean. Be concise, refer to Instagram and Meta policies when helpful (paraphrase).",
        "Twitter": "You are a Social Media Policy Consultant specializing in X (Twitter). Answer in Korean. Be concise, refer to X rules when helpful (paraphrase).",
    }
# 각 모드별 URL / 벡터스토어 / 임베딩 캐시
if "policy_urls" not in st.session_state:
    st.session_state.policy_urls = {
        # 필요시 원하시는 가이드 URL로 바꾸세요
        "Facebook": "https://transparency.fb.com/policies/ad-standards/",
        "Instagram": "https://transparency.fb.com/policies/ad-standards/",
        "Twitter": "https://business.x.com/en/help/policy-center/ads-policies.html",
    }
if "vstores" not in st.session_state:
    st.session_state.vstores: Dict[str, Optional[FAISS]] = {"Facebook": None, "Instagram": None, "Twitter": None}
if "embeddings" not in st.session_state:
    # 한 번만 로드 (메모리 캐시)
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------------------- 함수: 정책 페이지 로드/파싱/색인 ----------------------
def fetch_policy_text(url: str, timeout: int = 30) -> str:
    """정책 URL에서 본문 텍스트를 가져와 정제."""
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # 불필요한 스크립트/스타일 제거
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # 텍스트 추출
    text = soup.get_text(separator="\n")
    # 간단 정제
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]  # 빈 줄 제거
    return "\n".join(lines)

def build_vectorstore_from_text(text: str, source_url: str) -> FAISS:
    """텍스트를 chunk로 분할하고 FAISS 벡터스토어(메모리 상주)를 생성."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    chunks = splitter.split_text(text)

    docs = [
        Document(page_content=chunk, metadata={"source": source_url, "chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]
    vs = FAISS.from_documents(docs, st.session_state.embeddings)
    return vs

def ensure_vectorstore(mode: str, force_reload: bool = False) -> Optional[str]:
    """
    해당 모드의 벡터스토어가 비어 있으면 URL을 파싱해서 생성.
    force_reload=True면 항상 다시 구성.
    에러 메시지(있다면) 반환.
    """
    try:
        if force_reload or st.session_state.vstores.get(mode) is None:
            url = st.session_state.policy_urls.get(mode)
            if not url:
                return f"[오류] {mode} 모드의 정책 URL이 설정되어 있지 않습니다."
            with st.spinner(f"{mode} 광고 가이드 불러오는 중…"):
                text = fetch_policy_text(url)
                vs = build_vectorstore_from_text(text, url)
                st.session_state.vstores[mode] = vs
                st.toast(f"{mode} 정책 문서를 {len(vs.index_to_docstore_id)}개 청크로 색인 완료")
        return None
    except requests.RequestException as e:
        return f"[오류] 정책 페이지 요청 실패: {e}"
    except Exception as e:
        return f"[오류] 색인 생성 중 문제 발생: {e}"

def retrieve_context(query: str, mode: str, k: int = 5) -> str:
    """유저 쿼리에 맞게 상위 k개 청크를 검색하여 컨텍스트 텍스트로 합침."""
    vs = st.session_state.vstores.get(mode)
    if not vs:
        return ""
    try:
        docs = vs.similarity_search(query, k=k)
    except Exception:
        # 일부 버전/환경에서 검색 중 예외 대비
        docs = []
    if not docs:
        return ""
    context_blocks = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        idx = d.metadata.get("chunk_index", "?")
        context_blocks.append(f"[콘텍스트 {i} | {src} | chunk {idx}]\n{d.page_content}")
    return "\n\n".join(context_blocks)

# ---------------------- Ollama Chat ----------------------
def _build_system_prompt(mode: str) -> str:
    return st.session_state.system_prompts.get(mode, st.session_state.system_prompts["Facebook"]) + \
        "\n- Use bullet points when listing.\n- Ask for missing context only if essential.\n- Keep answers practical and actionable.\n- If provided, use the policy context below first."


def _to_ollama_messages(user_assistant_msgs, mode, rag_context=""):
    sys_prompt = _build_system_prompt(mode)
    if rag_context:
        sys_prompt += f"\n\n### Policy Context (RAG)\n{rag_context}\n\n" \
                      "지침: 위 'Policy Context'를 우선 근거로 활용하세요. 출처(URL)와 근거 문구를 한국어로 간단히 요약/인용(의역)하여 제시하세요."

    msgs = []
    # ✅ system → user로 변환
    msgs.append({"role": "user", "content": f"[시스템 지침]\n{sys_prompt}"})

    for m in user_assistant_msgs:
        msgs.append(m)

    return msgs

def ollama_chat_stream(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 1024,
) -> Iterator[str]:
    """/api/chat 스트리밍 제너레이터. Streamlit의 st.write_stream과 함께 사용."""
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "num_predict": int(max_tokens),
        },
    }
    try:
        with requests.post(url, json=payload, stream=True, timeout=(5, 600)) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = data.get("message", {})
                if "content" in msg:
                    yield msg["content"]
                if data.get("done"):
                    break
    except requests.RequestException as e:
        yield f"\n[오류] Ollama 서버 통신 실패: {e}"

def ollama_chat_once(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 1024,
) -> str:
    """스트리밍이 불가한 환경을 위한 단발성 호출."""
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "num_predict": int(max_tokens),
        },
    }
    try:
        r = requests.post(url, json=payload, timeout=(5, 120))
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "")
    except requests.RequestException as e:
        return f"[오류] Ollama 서버 통신 실패: {e}"

# ---------------------- 중앙 영역: UI ----------------------
st.title("🕊 Social Media Policy Consultant (with RAG)")

with st.sidebar:
    st.header("메뉴")
    st.caption("상담하고 싶은 SNS 종류를 바꿔 보세요. (버튼 클릭 시 해당 가이드를 크롤링/색인)")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Facebook", use_container_width=True):
            st.session_state.mode = "Facebook"
            err = ensure_vectorstore("Facebook")  # 색인 보장
            if err:
                st.error(err)
            else:
                st.toast("Facebook으로 전환 및 정책 색인 완료")
    with col2:
        if st.button("Instagram", use_container_width=True):
            st.session_state.mode = "Instagram"
            err = ensure_vectorstore("Instagram")
            if err:
                st.error(err)
            else:
                st.toast("Instagram으로 전환 및 정책 색인 완료")
    with col3:
        if st.button("Twitter", use_container_width=True):
            st.session_state.mode = "Twitter"
            err = ensure_vectorstore("Twitter")
            if err:
                st.error(err)
            else:
                st.toast("Twitter로 전환 및 정책 색인 완료")

    st.divider()
    if st.button("대화 초기화", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.success("대화를 초기화했습니다.")

    st.subheader("RAG 설정")
    st.caption("정책 URL을 자유롭게 교체할 수 있습니다.")
    for m in ["Facebook", "Instagram", "Twitter"]:
        st.session_state.policy_urls[m] = st.text_input(
            f"{m} 광고 가이드 URL",
            value=st.session_state.policy_urls[m],
            help=f"{m} 공식 광고 가이드(정책) 페이지 URL"
        )

    if st.button("현재 모드 정책 다시 불러오기(재색인)", use_container_width=True):
        msg = ensure_vectorstore(st.session_state.mode, force_reload=True)
        if msg:
            st.error(msg)
        else:
            st.success(f"{st.session_state.mode} 정책을 재색인했습니다.")

    st.subheader("Ollama 설정")
    ollama_url = st.text_input("Ollama Base URL", value="http://localhost:11434", help="예: http://localhost:11434")
    model_name = st.text_input("모델 이름", value="Phi3", help="ollama run 으로 받은 모델명")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.3, step=0.1)
    top_p = st.slider("top_p", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
    max_tokens = st.number_input("max_tokens (응답 최대 토큰)", min_value=1, max_value=8192, value=1024, step=64)

st.caption(f"현재 선택된 SNS: **{st.session_state.mode}**  |  모델: **{model_name}**  |  서버: {ollama_url}")

# 최초 진입 시 현재 모드 색인 보장
_ = ensure_vectorstore(st.session_state.mode)

# 기존 대화 렌더링
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 입력창
user_input = st.chat_input("메시지를 입력하세요…")

if user_input:
    # 유저 메시지 추가/표시
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- RAG: 현재 모드의 정책에서 관련 컨텍스트 검색 ---
    rag_context = retrieve_context(user_input, st.session_state.mode, k=3)
    print(f"rag_context: {rag_context}")

    # Ollama로 보낼 대화 메시지 구성 (컨텍스트를 system에 주입)
    ollama_msgs = _to_ollama_messages(st.session_state.messages, st.session_state.mode, rag_context=rag_context)
    print(f"ollama_msgs: {ollama_msgs}")

    # 어시스턴트 영역 생성 (스트리밍 출력)
    with st.chat_message("assistant"):
        try:
            chunks = ollama_chat_stream(
                base_url=ollama_url,
                model=model_name,
                messages=ollama_msgs,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            full_text = st.write_stream(chunks)
        except Exception:
            full_text = ollama_chat_once(
                base_url=ollama_url,
                model=model_name,
                messages=ollama_msgs,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            st.markdown(full_text)

    # 세션 저장
    st.session_state.messages.append({"role": "assistant", "content": full_text})

# ---------------------- 하단 도움말 ----------------------
with st.expander("ℹ️ 사용 팁"):
    st.markdown(
        """
        - 사이드바 SNS 버튼을 누르면 해당 광고 가이드(URL)를 크롤링/분할/벡터화하여 메모리 내 벡터스토어에 저장합니다.
        - 이후 유저 질문 시 유사도 검색으로 상위 근거를 찾아 **Policy Context (RAG)** 로 주입하므로 보다 정확한 정책 답변을 제공합니다.
        - URL은 자유롭게 수정 가능하며, “재색인” 버튼으로 즉시 반영됩니다.
        - 임베딩 모델은 `sentence-transformers/all-MiniLM-L6-v2`, 벡터스토어는 메모리 상주 FAISS를 사용합니다.
        - Ollama 서버가 로컬에서 실행 중인지 확인하세요: `ollama serve` 후 `ollama run {모델}`
        - 응답이 없으면 모델명, 서버 URL, 방화벽/프록시 여부를 확인하세요.
        - 장문 요약/분석 등은 `max_tokens`를 늘려 보세요.
        """
    )
