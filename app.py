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
PDF_PATHS = [
    DATA_DIR / "빅데이터암기키트.pdf",
    DATA_DIR / "빅분기_필기_요약정리.pdf"
]

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
    """여러 PDF 텍스트 로드 및 병합"""
    try:
        from pypdf import PdfReader
        all_text = ""
        for pdf_path in PDF_PATHS:
            if pdf_path.exists():
                reader = PdfReader(str(pdf_path))
                for page in reader.pages:
                    all_text += page.extract_text() + "\n"
                all_text += "\n---\n"  # PDF 구분자
        return all_text
    except:
        pass
    return ""

def get_subject_number(topic: str) -> int:
    """과목명에서 과목 번호 추출"""
    if "1과목" in topic:
        return 1
    elif "2과목" in topic:
        return 2
    elif "3과목" in topic:
        return 3
    elif "4과목" in topic:
        return 4
    return 0  # 전체

def get_sample_questions_by_subject(questions, subject_num, n=5):
    """해당 과목의 기출문제만 필터링하여 샘플링"""
    if subject_num == 0:
        filtered = questions
    else:
        filtered = [q for q in questions if q.get('subject') == subject_num]

    if not filtered:
        filtered = questions

    samples = random.sample(filtered, min(n, len(filtered)))
    formatted = []
    for q in samples:
        q_type = classify_question_type(q['question'])
        formatted.append(f"""
[유형: {q_type}]
문제: {q['question']}
1. {q['choices'][0]}
2. {q['choices'][1]}
3. {q['choices'][2]}
4. {q['choices'][3]}
정답: {q['answer']}번
키워드: {', '.join(q.get('keywords', []))}
""")
    return "\n---\n".join(formatted)

def classify_question_type(question_text: str) -> str:
    """문제 유형 분류"""
    if any(x in question_text for x in ["아닌 것", "옳지 않은", "틀린 것", "해당하지 않는"]):
        return "부정형"
    elif any(x in question_text for x in ["모두 고르", "모두 선택", "해당하는 것을 모두"]):
        return "나열형"
    elif any(x in question_text for x in ["차이점", "비교", "구분", "다른 점"]):
        return "비교형"
    elif any(x in question_text for x in ["계산", "구하", "얼마", "몇 "]):
        return "계산형"
    else:
        return "정의형"

def analyze_question_type_distribution(questions, subject_num=0):
    """과목별 문제 유형 분포 분석"""
    if subject_num == 0:
        filtered = questions
    else:
        filtered = [q for q in questions if q.get('subject') == subject_num]

    type_count = {"정의형": 0, "부정형": 0, "비교형": 0, "나열형": 0, "계산형": 0}
    for q in filtered:
        q_type = classify_question_type(q['question'])
        type_count[q_type] += 1

    total = len(filtered)
    if total == 0:
        return type_count

    return {k: f"{v}개 ({v/total*100:.0f}%)" for k, v in type_count.items()}

def get_sample_questions(questions, n=5):
    """샘플 기출문제 가져오기 (하위 호환)"""
    return get_sample_questions_by_subject(questions, 0, n)

def extract_theory_by_subject(theory_text: str, subject_num: int) -> str:
    """과목별로 암기키트 내용 추출"""
    subject_keywords = {
        1: ["분석 기획", "CRISP-DM", "분석 마스터플랜", "분석 거버넌스", "데이터 거버넌스"],
        2: ["EDA", "탐색적", "시각화", "데이터 탐색", "상관분석", "분포"],
        3: ["회귀", "분류", "군집", "모델링", "머신러닝", "알고리즘", "앙상블"],
        4: ["혼동행렬", "ROC", "AUC", "정밀도", "재현율", "결과해석", "F1"]
    }

    if subject_num == 0 or subject_num not in subject_keywords:
        return theory_text[:4000]

    keywords = subject_keywords[subject_num]
    lines = theory_text.split('\n')
    relevant_lines = []

    for line in lines:
        if any(kw in line for kw in keywords):
            relevant_lines.append(line)

    result = '\n'.join(relevant_lines)
    if len(result) < 500:
        return theory_text[:4000]

    return result[:4000]

def calculate_similarity(text1: str, text2: str) -> float:
    """두 문장의 유사도 계산 (간단한 키워드 매칭)"""
    words1 = set(text1.replace("?", "").replace(".", "").split())
    words2 = set(text2.replace("?", "").replace(".", "").split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0

def filter_duplicate_questions(new_questions: list, existing_questions: list, threshold=0.7) -> list:
    """기출과 유사한 문제 제거"""
    filtered = []
    for new_q in new_questions:
        is_duplicate = False
        for exist_q in existing_questions:
            if calculate_similarity(new_q.get('question', ''), exist_q.get('question', '')) > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            filtered.append(new_q)
    return filtered

def generate_questions_batch(topic: str, n_questions: int,
                             exam_samples: str, theory_text: str) -> list:
    """단일 배치로 문제 생성"""
    client = OpenAI()

    subject_num = get_subject_number(topic)
    theory_excerpt = extract_theory_by_subject(theory_text, subject_num)

    prompt = f"""당신은 빅데이터분석기사 시험 출제위원입니다.

## 출제 규칙
1. 기출문제와 동일하거나 유사한 문제는 절대 금지
2. 기출 스타일(문장 구조, 보기 형식)은 유지
3. 오답은 정답과 혼동되기 쉽게 설계
4. 각 과목의 핵심 키워드 반드시 포함

## 문제 유형별 출제 비율
- 정의형 30% (개념, 정의를 묻는 문제)
- 부정형 20% (옳지 않은 것, 아닌 것)
- 비교형 20% (차이점, 구분)
- 나열형 15% (해당하는 것 모두)
- 계산형 15% (수치 계산)

## 오답 설계 규칙
1. 정답과 비슷한 용어 사용 (예: 무결성 vs 완전성 vs 정확성)
2. 숫자/순서 변형 (예: 3V vs 4V vs 5V)
3. 관련 개념이지만 틀린 답 사용
4. 그럴듯하지만 미묘하게 틀린 설명

## 기출문제 예시 (스타일 참고용)
{exam_samples}

## 이론 내용
{theory_excerpt}

## 요청
- 과목: {topic}
- 문제 수: {n_questions}개
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
        subject_num = get_subject_number(subj)
        samples = get_sample_questions_by_subject(exam_questions, subject_num, 5)
        response = generate_questions_batch(subj, n, samples, theory_text)
        parsed = parse_generated_questions(response)
        # 기출과 중복되는 문제 제거
        filtered = filter_duplicate_questions(parsed, exam_questions)
        return filtered

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

def render_past_exam_mode(exam_questions):
    """기출문제 풀기 모드 렌더링"""

    # 문제 선택 전
    if not st.session_state.past_exam_questions:
        st.subheader("기출문제 풀기")

        # 필터 옵션
        col1, col2 = st.columns(2)

        with col1:
            subject_filter = st.selectbox(
                "과목 선택",
                ["전체", "1과목: 빅데이터 분석 기획", "2과목: 빅데이터 탐색",
                 "3과목: 빅데이터 모델링", "4과목: 빅데이터 결과해석"]
            )

        with col2:
            mode = st.selectbox(
                "풀이 방식",
                ["순서대로", "랜덤 섞기"]
            )

        # 과목 필터링
        subject_num = get_subject_number(subject_filter)
        if subject_num == 0:
            filtered_questions = exam_questions
        else:
            filtered_questions = [q for q in exam_questions if q.get('subject') == subject_num]

        st.write(f"선택된 문제: {len(filtered_questions)}개")

        # 문제 수 선택
        n_questions = st.slider("풀 문제 수", min_value=5, max_value=min(80, len(filtered_questions)),
                                value=min(20, len(filtered_questions)), step=5)

        if st.button("문제 풀기 시작", type="primary"):
            selected = filtered_questions[:n_questions] if mode == "순서대로" else random.sample(filtered_questions, n_questions)
            st.session_state.past_exam_questions = selected
            st.session_state.current_q_index = 0
            st.session_state.user_answers = {}
            st.session_state.show_results = False
            st.rerun()

        return

    # 문제 풀이 중
    questions = st.session_state.past_exam_questions
    total_q = len(questions)
    answered = len(st.session_state.user_answers)

    # 진행률 표시
    correct_so_far = sum(1 for i, q in enumerate(questions)
                         if str(i) in st.session_state.user_answers
                         and st.session_state.user_answers[str(i)] == q['answer'])
    wrong_so_far = answered - correct_so_far

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("진행률", f"{answered}/{total_q}")
    with col2:
        st.metric("맞은 문제", f"{correct_so_far}개")
    with col3:
        accuracy = (correct_so_far / answered * 100) if answered > 0 else 0
        st.metric("정답률", f"{accuracy:.0f}%")

    st.divider()

    # 결과 화면
    if st.session_state.show_results:
        st.subheader("최종 결과")

        score = correct_so_far / total_q * 100
        delta = "합격" if score >= 60 else "불합격"
        st.metric("최종 점수", f"{score:.0f}점", delta)

        st.divider()

        # 오답 노트
        st.subheader("오답 노트")
        for i, q in enumerate(questions):
            user_ans = st.session_state.user_answers.get(str(i))
            if user_ans != q['answer']:
                with st.expander(f"[X] 문제 {i+1}: {q['question'][:50]}..."):
                    st.markdown(f"**{q['question']}**")
                    for j, choice in enumerate(q['choices'], 1):
                        if j == q['answer']:
                            st.success(f"{j}. {choice} (정답)")
                        elif j == user_ans:
                            st.error(f"{j}. {choice} (선택)")
                        else:
                            st.write(f"{j}. {choice}")
                    if q.get('keywords'):
                        st.caption(f"키워드: {', '.join(q['keywords'])}")

        if st.button("다시 풀기", type="primary"):
            st.session_state.past_exam_questions = []
            st.session_state.user_answers = {}
            st.session_state.show_results = False
            st.rerun()

        return

    # 현재 문제 표시
    q_idx = st.session_state.current_q_index
    q = questions[q_idx]

    st.subheader(f"문제 {q_idx + 1} / {total_q}")
    st.markdown(f"**{q['question']}**")

    current_answer = st.session_state.user_answers.get(str(q_idx))
    answered_this = current_answer is not None

    # 보기 표시
    for i, choice in enumerate(q['choices'], 1):
        is_selected = (current_answer == i)
        is_correct_choice = (i == q['answer'])

        if answered_this:
            if is_correct_choice:
                st.success(f"{i}. {choice} [정답]")
            elif is_selected and not is_correct_choice:
                st.error(f"{i}. {choice} [오답]")
            else:
                st.write(f"{i}. {choice}")
        else:
            if st.button(f"{i}. {choice}", key=f"past_choice_{q_idx}_{i}", use_container_width=True):
                st.session_state.user_answers[str(q_idx)] = i
                st.rerun()

    # 정답 선택 후 피드백
    if answered_this:
        is_correct = current_answer == q['answer']
        if is_correct:
            st.success("정답입니다!")
        else:
            st.error(f"오답입니다. 정답은 {q['answer']}번입니다.")

        if q.get('keywords'):
            st.caption(f"키워드: {', '.join(q['keywords'])}")

    # 네비게이션
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if q_idx > 0:
            if st.button("이전 문제", use_container_width=True):
                st.session_state.current_q_index -= 1
                st.rerun()

    with col2:
        if answered == total_q:
            if st.button("최종 결과 보기", type="primary", use_container_width=True):
                st.session_state.show_results = True
                st.rerun()

    with col3:
        if q_idx < total_q - 1:
            if st.button("다음 문제", use_container_width=True):
                st.session_state.current_q_index += 1
                st.rerun()


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

        # answer 값 검증 및 수정
        for q in questions:
            answer = q.get('answer', 1)
            # answer가 0이면 1로, 5 이상이면 4로 수정
            if answer < 1:
                q['answer'] = 1
            elif answer > 4:
                q['answer'] = 4
            # answer가 문자열이면 정수로 변환
            if isinstance(q.get('answer'), str):
                try:
                    q['answer'] = int(q['answer'])
                except:
                    q['answer'] = 1

            # 보기에서 A., B., C., D. 또는 1., 2., 3., 4. 접두사 제거
            if 'choices' in q:
                cleaned_choices = []
                for choice in q['choices']:
                    # A. B. C. D. 또는 1. 2. 3. 4. 제거
                    choice = str(choice).strip()
                    if len(choice) > 2 and choice[0] in 'ABCD1234' and choice[1] in '.)':
                        choice = choice[2:].strip()
                    elif len(choice) > 3 and choice[:2] in ['A.', 'B.', 'C.', 'D.', '1.', '2.', '3.', '4.']:
                        choice = choice[2:].strip()
                    cleaned_choices.append(choice)
                q['choices'] = cleaned_choices

        return questions
    except Exception as e:
        print(f"JSON 파싱 오류: {e}")
        return []

def main():
    st.set_page_config(
        page_title="BigData-Pass AI",
        layout="wide"
    )

    st.title("BigData-Pass AI")

    # Session State 초기화
    if "mode" not in st.session_state:
        st.session_state.mode = None  # None, "ai_generate", "past_exam"
    if "generated_questions" not in st.session_state:
        st.session_state.generated_questions = []
    if "current_q_index" not in st.session_state:
        st.session_state.current_q_index = 0
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}
    if "show_results" not in st.session_state:
        st.session_state.show_results = False
    if "past_exam_questions" not in st.session_state:
        st.session_state.past_exam_questions = []

    # 데이터 로드
    exam_questions = load_exam_questions()
    theory_text = load_pdf_text()

    # 모드 선택 화면
    if st.session_state.mode is None:
        st.subheader("학습 모드 선택")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### AI 문제 생성")
            st.write("AI가 기출문제 스타일로 새로운 문제를 생성합니다.")
            st.write(f"학습 데이터: 기출 {len(exam_questions)}문제")
            if st.button("AI 문제 생성 모드", type="primary", use_container_width=True):
                st.session_state.mode = "ai_generate"
                st.rerun()

        with col2:
            st.markdown("### 기출문제 풀기")
            st.write("실제 기출문제를 한 문제씩 풀어봅니다.")
            st.write(f"총 {len(exam_questions)}문제 보유")
            if st.button("기출문제 풀기 모드", type="secondary", use_container_width=True):
                st.session_state.mode = "past_exam"
                st.rerun()

        return

    # 뒤로가기 버튼
    if st.button("< 모드 선택으로 돌아가기"):
        st.session_state.mode = None
        st.session_state.generated_questions = []
        st.session_state.past_exam_questions = []
        st.session_state.user_answers = {}
        st.session_state.show_results = False
        st.session_state.current_q_index = 0
        st.rerun()

    st.divider()

    # 기출문제 풀기 모드
    if st.session_state.mode == "past_exam":
        render_past_exam_mode(exam_questions)
        return

    # AI 문제 생성 모드
    if not st.session_state.generated_questions:
        st.subheader("AI 문제 생성 설정")

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
