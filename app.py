import streamlit as st
import json
import random
import concurrent.futures
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 설정
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
JSON_DIR = DATA_DIR / "json"
PDF_PATH = DATA_DIR / "빅데이터암기키트.pdf"

# 데이터 로드
@st.cache_data
def load_exam_questions():
    """기출문제 JSON 로드"""
    questions = []
    json_files = sorted(JSON_DIR.glob("exam_*.json"))

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for q in data['questions']:
                questions.append(q)

    return questions

@st.cache_data
def load_pdf_text():
    """PDF 텍스트 로드"""
    try:
        from pypdf import PdfReader
        if PDF_PATH.exists():
            reader = PdfReader(str(PDF_PATH))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except:
        pass
    return ""

def get_sample_questions(questions, n=5):
    """샘플 기출문제 가져오기"""
    samples = random.sample(questions, min(n, len(questions)))
    formatted = []
    for q in samples:
        formatted.append(f"""
문제: {q['question']}
1. {q['choices'][0]}
2. {q['choices'][1]}
3. {q['choices'][2]}
4. {q['choices'][3]}
정답: {q['answer']}번
키워드: {', '.join(q.get('keywords', []))}
""")
    return "\n---\n".join(formatted)

def generate_questions_batch(topic: str, n_questions: int,
                             exam_samples: str, theory_text: str) -> list:
    """단일 배치로 문제 생성"""
    client = OpenAI()
    theory_excerpt = theory_text[:4000] if len(theory_text) > 4000 else theory_text

    prompt = f"""당신은 빅데이터분석기사 시험 출제위원입니다.

## 기출문제 예시
{exam_samples}

## 이론 내용
{theory_excerpt}

## 요청
- 주제: {topic}
- 문제 수: {n_questions}개
- 새로운 문제 생성
- 해설은 간단히 1-2문장으로

## JSON 형식으로만 출력
[{{"question":"문제","choices":["1","2","3","4"],"answer":1,"keywords":["키워드"],"explanation":"해설"}}]
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def generate_questions_parallel(topic: str, n_questions: int,
                                 exam_questions: list, theory_text: str,
                                 progress_callback=None) -> list:
    """병렬로 문제 생성 (더 빠름)"""
    all_questions = []

    if "전체" in topic:
        # 4과목 병렬 생성 (각 10문제씩 2배치 = 총 80문제)
        subjects = [
            ("1과목: 빅데이터 분석 기획", 10),
            ("1과목: 빅데이터 분석 기획", 10),
            ("2과목: 빅데이터 탐색", 10),
            ("2과목: 빅데이터 탐색", 10),
            ("3과목: 빅데이터 모델링", 10),
            ("3과목: 빅데이터 모델링", 10),
            ("4과목: 빅데이터 결과해석", 10),
            ("4과목: 빅데이터 결과해석", 10)
        ]
    else:
        # 단일 과목은 10문제씩 2배치로
        subjects = [(topic, 10), (topic, 10)]

    def generate_one(args):
        subj, n = args
        samples = get_sample_questions(exam_questions, 5)
        response = generate_questions_batch(subj, n, samples, theory_text)
        return parse_generated_questions(response)

    # 병렬 실행
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(generate_one, s) for s in subjects]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            if result:
                all_questions.extend(result)
            if progress_callback:
                progress_callback(i + 1, len(subjects))

    return all_questions

def parse_generated_questions(response: str):
    """응답에서 JSON 파싱"""
    try:
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0]
        else:
            json_str = response

        json_str = json_str.strip()

        # 잘린 JSON 복구 시도
        if not json_str.endswith("]"):
            # 마지막 완전한 객체까지만 파싱
            last_complete = json_str.rfind("}")
            if last_complete > 0:
                json_str = json_str[:last_complete + 1] + "]"

        questions = json.loads(json_str)
        return questions
    except Exception as e:
        print(f"JSON 파싱 오류: {e}")
        return []

def main():
    st.set_page_config(
        page_title="BigData-Pass AI",
        layout="wide"
    )

    st.title("BigData")

    # Session State 초기화
    if "generated_questions" not in st.session_state:
        st.session_state.generated_questions = []
    if "current_q_index" not in st.session_state:
        st.session_state.current_q_index = 0
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}
    if "show_results" not in st.session_state:
        st.session_state.show_results = False

    # 데이터 로드
    exam_questions = load_exam_questions()
    theory_text = load_pdf_text()

    # 문제 생성 영역
    if not st.session_state.generated_questions:
        st.subheader("문제 생성 설정")

        topic = st.selectbox(
            "출제 과목",
            ["전체 (80문제)", "1과목: 빅데이터 분석 기획 (20문제)",
             "2과목: 빅데이터 탐색 (20문제)", "3과목: 빅데이터 모델링 (20문제)",
             "4과목: 빅데이터 결과해석 (20문제)"]
        )

        # 과목에 따라 문제 수 자동 설정
        if "전체" in topic:
            n_questions = 80
        else:
            n_questions = 20

        st.write(f"기출문제 {len(exam_questions)}개, 암기키트 {len(theory_text)}자 로드됨")
        st.write(f"생성할 문제 수: {n_questions}개")

        if st.button("문제 생성", type="primary"):
            progress_bar = st.progress(0, text="AI가 문제를 생성 중입니다...")

            def update_progress(current, total):
                progress_bar.progress(current / total, text=f"생성 중... ({current}/{total} 배치 완료)")

            try:
                questions = generate_questions_parallel(
                    topic=topic,
                    n_questions=n_questions,
                    exam_questions=exam_questions,
                    theory_text=theory_text,
                    progress_callback=update_progress
                )

                if questions:
                    st.session_state.generated_questions = questions
                    st.session_state.current_q_index = 0
                    st.session_state.user_answers = {}
                    st.session_state.show_results = False
                    st.rerun()
                else:
                    st.error("문제 생성 실패")
            except Exception as e:
                st.error(f"오류 발생: {e}")

    # 문제 풀이 영역
    else:
        questions = st.session_state.generated_questions
        total_q = len(questions)

        # 진행률
        answered = len(st.session_state.user_answers)
        st.progress(answered / total_q, text=f"진행률: {answered}/{total_q}")

        # 문제 네비게이션
        cols = st.columns(min(10, total_q))
        for i in range(min(10, total_q)):
            with cols[i]:
                is_current = (i == st.session_state.current_q_index)
                is_answered = str(i) in st.session_state.user_answers

                label = f"{i+1}"
                if is_answered:
                    label += " [v]"

                if st.button(label, key=f"nav_{i}",
                           type="primary" if is_current else "secondary",
                           use_container_width=True):
                    st.session_state.current_q_index = i
                    st.rerun()

        st.divider()

        # 현재 문제 표시
        if not st.session_state.show_results:
            q_idx = st.session_state.current_q_index
            q = questions[q_idx]

            st.subheader(f"문제 {q_idx + 1}")
            st.markdown(f"**{q['question']}**")

            # 보기 선택
            current_answer = st.session_state.user_answers.get(str(q_idx))
            answered_this = current_answer is not None

            for i, choice in enumerate(q['choices'], 1):
                is_selected = (current_answer == i)
                is_correct_choice = (i == q['answer'])

                # 이미 답을 선택한 경우 정답/오답 표시
                if answered_this:
                    if is_correct_choice:
                        st.success(f"{i}. {choice} [정답]")
                    elif is_selected and not is_correct_choice:
                        st.error(f"{i}. {choice} [오답]")
                    else:
                        st.write(f"{i}. {choice}")
                else:
                    # 아직 답을 선택하지 않은 경우
                    if st.button(
                        f"{i}. {choice}",
                        key=f"choice_{q_idx}_{i}",
                        use_container_width=True
                    ):
                        st.session_state.user_answers[str(q_idx)] = i
                        st.rerun()

            # 정답 선택 후 해설 표시
            if answered_this:
                is_correct = current_answer == q['answer']
                if is_correct:
                    st.success("정답입니다!")
                else:
                    st.error(f"오답입니다. 정답은 {q['answer']}번입니다.")

                if q.get('explanation'):
                    st.info(f"해설: {q['explanation']}")

                if q.get('keywords'):
                    st.caption(f"키워드: {', '.join(q['keywords'])}")

            # 네비게이션
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                if q_idx > 0:
                    if st.button("이전", use_container_width=True):
                        st.session_state.current_q_index -= 1
                        st.rerun()

            with col2:
                if answered == total_q:
                    if st.button("최종 결과 보기", type="primary", use_container_width=True):
                        st.session_state.show_results = True
                        st.rerun()

            with col3:
                if q_idx < total_q - 1:
                    if st.button("다음", use_container_width=True):
                        st.session_state.current_q_index += 1
                        st.rerun()

        else:
            # 채점 결과
            st.subheader("채점 결과")

            correct_count = 0
            for i, q in enumerate(questions):
                user_ans = st.session_state.user_answers.get(str(i))
                if user_ans == q['answer']:
                    correct_count += 1

            score = correct_count / total_q * 100

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("맞은 문제", f"{correct_count}개")
            with col2:
                st.metric("틀린 문제", f"{total_q - correct_count}개")
            with col3:
                delta = "합격" if score >= 60 else "불합격"
                st.metric("점수", f"{score:.0f}점", delta)

            st.divider()

            # 문제별 결과
            for i, q in enumerate(questions):
                user_ans = st.session_state.user_answers.get(str(i))
                is_correct = user_ans == q['answer']

                icon = "[O]" if is_correct else "[X]"

                with st.expander(f"{icon} 문제 {i+1}: {q['question'][:50]}..."):
                    st.markdown(f"**{q['question']}**")

                    for j, choice in enumerate(q['choices'], 1):
                        if j == q['answer']:
                            st.markdown(f"**{j}. {choice}** (정답)")
                        elif j == user_ans:
                            st.markdown(f"~~{j}. {choice}~~ (선택)")
                        else:
                            st.write(f"{j}. {choice}")

                    if q.get('explanation'):
                        st.info(f"해설: {q['explanation']}")

                    if q.get('keywords'):
                        st.caption(f"키워드: {', '.join(q['keywords'])}")

            st.divider()

            if st.button("새로운 문제 생성하기", type="primary"):
                st.session_state.generated_questions = []
                st.session_state.show_results = False
                st.rerun()

if __name__ == "__main__":
    main()
