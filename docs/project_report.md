# تقرير سير العمل والاختلافات في الكود (Project Workflow & Code Differences Report)

## سبب الاختلافات والتكرار في الكود
السبب بسيط: **Member 2 و Member 3 اشتغلوا كل واحد لوحده من غير ما يتفقوا مع بعض.**

Member 3 كان محتاج يجرّب الكود بتاعه (الـ Loop والـ Cache)، بس كود Member 2 مكانش جاهز وقتها. فـ Member 3 عمل حاجة اسمها **Stubs** (يعني نسخ مؤقتة بديلة) عشان يقدر يشغّل ويجرّب كوده لوحده. الـ Stubs دي فيها Generator و Evaluator و Memory و Config كلها مكتوبة من الصفر عشان Member 3 يشتغل.

بعد كده لما Member 2 خلّص كوده الحقيقي، محدش رجع ودمج الاتنين مع بعض. فدلوقتي عندنا نسختين من نفس الحاجة.

---

## خطوات عمل النظام بالترتيب

### الخطوة 1: المستخدم يرفع ملفاته (شغلك أنتِ)
يقوم المستخدم برفع الملفات المراد البحث فيها.

### الخطوة 2: المستخدم يسأل سؤال
بيكتب في الشات: *"ما هي مميزات الذكاء الاصطناعي؟"*

### الخطوة 3: البحث عن الفقرات المتعلقة (Retrieval)
النظام بياخد السؤال، يحوّله لأرقام (Embedding)، ويبحث في ChromaDB عن أقرب 5 فقرات في المعنى.

⚠️ **هنا أول اختلاف بين Member 2 و Member 3:**

**Member 2** كتب كوده إنه يستقبل الـ Context كـ **نص واحد طويل (string)**:
```python
# كود Member 2 — run_generator بتاخد context كـ string
def run_generator(question: str, context: str, feedback: str = "None") -> str:
```

**Member 3** كتب كوده إنه يتعامل مع الـ Context كـ **قائمة فقرات (List[str])**:
```python
# كود Member 3 — الـ Retriever بيرجّع List[str]
class Retriever(Protocol):
    def get_relevant_documents(self, query: str) -> List[str]:
```

**الحل:** في ملف `adapters.py` هنعمل تحويل بسيط:
```python
# بنلزّق الفقرات في نص واحد قبل ما نبعتها لكود Member 2
context_str = "\n\n".join(context_list)
```

---

### الخطوة 4: المُوَلِّد يكتب إجابة (Generator)
المُوَلِّد بياخد السؤال + الفقرات المتعلقة + أي ملاحظات (Feedback) من المُقيّم لو دي مش أول محاولة.

**كود Member 2 (الشغّال):**
```python
# في prompts.py — التعليمات اللي بتتبعت للنموذج
GENERATOR_SYSTEM_PROMPT = """
You are a precise, grounded AI assistant.
1. Answer using ONLY the provided retrieved context.
2. Do not invent or hallucinate facts.
3. If context doesn't have the answer, say: "The required information 
   is not available in the provided sources."
4. If Evaluator Feedback is provided, revise your answer accordingly.
Retrieved Context: {context}
Evaluator Feedback (if any): {feedback}
"""
```
```python
# في llm_core.py — بينادي النموذج ويحفظ في الذاكرة
def run_generator(question, context, feedback="None"):
    answer = generator_chain.invoke({
        "question": question,
        "context": context,
        "feedback": feedback,
        "generator_history": history    # ← هنا بيبعت الذاكرة
    })
    # بيحفظ السؤال والإجابة في الذاكرة
    generator_memory.save_context(
        {"question": f"Question: {question} (Feedback: {feedback})"},
        {"output": answer}
    )
    return answer
```

---

### الخطوة 5: المُقَيِّم يراجع الإجابة (Evaluator)
**كود Member 2:**
```python
EVALUATOR_SYSTEM_PROMPT = """
You are an objective AI Quality Assurance Evaluator.
Evaluation Criteria:
1. Accuracy & Grounding: Is every claim backed by the context?
2. Relevance: Does the answer address the question?
3. Completeness: Did it cover all parts of the question?
4. Hallucination Check: Any claims not in the context?
"""
```
المُقيّم بيرجّع نتيجة فيها حاجتين: هل الإجابة كويسة؟ وإيه الملاحظات؟

⚠️ **هنا تاني اختلاف بين Member 2 و Member 3:**

**Member 2** سمّى النتيجة كده:
```python
class EvaluationResult(BaseModel):
    is_satisfactory: bool    # ← اسم الخاصية
    feedback: str
```

**Member 3** سمّاها كده:
```python
class EvaluationResult:
    is_acceptable: bool      # ← اسم مختلف!
    feedback: str
    score: float             # ← حاجة زيادة مش موجودة عند Member 2
```

**الحل:** في `adapters.py`:
```python
# بناخد نتيجة Member 2 ونحوّلها لشكل Member 3
result_m3 = EvaluationResult(
    is_acceptable=result_m2.is_satisfactory,   # بنغيّر الاسم بس
    feedback=result_m2.feedback,
)
```

---

### الخطوة 6: الذاكرة (Memory) — إيه دورها بالظبط؟

#### ليه محتاجين ذاكرة أصلاً؟
تخيلي إن المُوَلِّد كتب إجابة، والمُقيّم قاله "الإجابة ناقصة، ضيف معلومات عن النقطة الثالثة". المُوَلِّد لازم يفتكر إجابته الأولى والملاحظات اللي جاتله عشان يقدر يحسّنها. من غير ذاكرة، كل مرة هيبدأ من الصفر كأنه أول مرة يشوف السؤال ده.

**ذاكرة المُوَلِّد (Generator Memory) بتحفظ:**
- السؤال اللي اتسأل
- الإجابة اللي كتبها
- الملاحظات (Feedback) اللي جاتله من المُقيّم
- الإجابة المعدّلة بعد الملاحظات

**ذاكرة المُقيّم (Evaluator Memory) بتحفظ:**
- الإجابات اللي قيّمها قبل كده
- الأحكام اللي طلّعها (قبل/رفض)
- الملاحظات اللي كتبها

#### ليه لازم الذاكرتين منفصلتين؟
لأن المُوَلِّد والمُقيّم لازم يشتغلوا باستقلالية. لو المُوَلِّد يقدر يشوف ذاكرة المُقيّم، ممكن "يغش" ويكتب إجابة مصمّمة عشان تعجب المُقيّم مش عشان تكون صح فعلاً. والعكس، لو المُقيّم يشوف ذاكرة المُوَلِّد، ممكن يتحيّز.

⚠️ **هنا تالت اختلاف كبير بين Member 2 و Member 3:**

**Member 2** عمل الذاكرة كـ **متغيرين ثابتين في الكود (module-level singletons):**
```python
# في memories.py — ذاكرة واحدة بس للنظام كله
generator_memory = ConversationBufferMemory(
    memory_key="generator_history",
    return_messages=True
)
evaluator_memory = ConversationBufferMemory(
    memory_key="evaluator_history",
    return_messages=True
)
```
*المشكلة هنا:* لو مستخدمين اتنين سألوا في نفس الوقت، الاتنين هيشاركوا نفس الذاكرة. يعني ذاكرة محمد هتتلخبط مع ذاكرة سارة.

**Member 3** عمل الذاكرة **لكل مستخدم لوحده عن طريق `session_id`:**
```python
# في memory.py — كل مستخدم له ذاكرة خاصة
class GeneratorMemory:
    _prefix = "mem:generator"
    def __init__(self, session_id: str):   # ← كل مستخدم له ID مختلف
        self._key = f"mem:generator:{session_id}"
    def add_turn(self, question, context, answer):
        # بيحفظ في Redis تحت مفتاح خاص بالمستخدم ده
        history = self._load()
        history.append({"question": question, "answer": answer})
        self._save(history)
```
وكمان لو Redis مش شغّال، بيشتغل عادي بذاكرة محلية (fallback). يعني النظام مش بيقع.

**الحل:** هنستخدم ذاكرة Member 3 لأنها أحسن وأأمن.

---

### الخطوة 7: حلقة التحسين (Feedback Loop — شغل Member 3)
ده قلب المشروع. Member 3 كتبه في `orchestrator.py` باستخدام LCEL (وهو أسلوب من LangChain لربط خطوات الذكاء الاصطناعي ببعض):
```python
# الخطوات الثلاثة مربوطة ببعض بـ LCEL
self.single_pass_chain = (
    self._retrieve_step    # الخطوة 1: جيب الفقرات المتعلقة
    | self._generate_step  # الخطوة 2: المُوَلِّد يكتب إجابة
    | self._evaluate_step  # الخطوة 3: المُقيّم يراجعها
)
```

واللوب نفسه:
```python
def run(self, question):
    for iteration in range(1, max_iterations + 1):   # أقصى حد 4 مرات
        state = self.single_pass_chain.invoke(state)  # شغّل الـ 3 خطوات
        if state["evaluation"].is_acceptable:          # المُقيّم وافق؟
            return final_answer                        # ✅ ابعت الإجابة
        state["feedback"] = state["evaluation"].feedback  # ❌ ابعت الملاحظات للمُوَلِّد
    # لو وصلنا هنا يبقى الـ 4 محاولات خلصوا من غير موافقة
    return best_answer + "⚠️ تنبيه: لم يتم التحقق بالكامل"
```

---

### الخطوة 8: التخزين المؤقت (Redis Cache)
كل خطوة من الخطوات الثلاثة (بحث، توليد، تقييم) بيتم تخزين نتيجتها في Redis:
```python
# قبل ما يبحث — بيشوف الأول لو نفس السؤال اتسأل قبل كده
cached = cache.get_retrieval(question)
if cached is not None:
    return cached              # ← مش محتاج يبحث تاني، يرجّع النتيجة المحفوظة
# لو مش موجود في الكاش — يبحث عادي ويحفظ النتيجة
context = retriever.get_relevant_documents(question)
cache.set_retrieval(question, context)   # ← يحفظها عشان المرة الجاية
```

---

## ملخص الاختلافات بين الأعضاء

| الموضوع | Member 2 | Member 3 | الأفضل للاستخدام |
|---|---|---|---|
| **الإعدادات (Config)** | ملف واحد، بيقع لو مفيش API Key | ملفات منظمة بـ dataclasses، بيتعامل مع الأخطاء | **Member 3** |
| **الذاكرة (Memory)** | ذاكرة واحدة مشتركة لكل الناس | ذاكرة خاصة لكل مستخدم + محفوظة في Redis | **Member 3** |
| **شكل الـ Context** | نص واحد (`str`) | قائمة فقرات (`List[str]`) | مفيش أحسن، بس لازم نوحّدهم |
| **اسم النتيجة** | `is_satisfactory` | `is_acceptable` | نفس المعنى، لازم نترجم بينهم |
| **الـ LLM** | نموذج واحد ثابت | Factory function بتعمل نموذج جديد كل مرة | **Member 3** (أكثر مرونة) |
