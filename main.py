import os
import joblib
import pandas as pd
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

app = FastAPI(title="Math Score Predictor")

# Шлях до твоєї збереженої моделі
MODEL_PATH = "best_math_predict_model.pkl"

# Завантажуємо модель один раз при запуску сервера
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"Файл моделі {MODEL_PATH} не знайдено! Спочатку запусти ML-скрипт навчання.")

# Головна (і єдина) сторінка додатка
@app.get("/", response_class=HTMLResponse)
async def index():
    return get_html_template()

# Обробка даних з форми та повернення результату
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    gender: str = Form(...),
    race: str = Form(...),
    education: str = Form(...),
    lunch: str = Form(...),
    course: str = Form(...),
    reading_score: int = Form(...),
    writing_score: int = Form(...)
):
    # 1. Формуємо DataFrame з точно такими ж назвами колонок, як у початковому X
    input_data = pd.DataFrame([{
        'gender': gender,
        'race/ethnicity': race,
        'parental level of education': education,
        'lunch': lunch,
        'test preparation course': course,
        'reading score': reading_score,
        'writing score': writing_score
    }])
    
    # 2. Робимо прогноз через завантажений пайплайн (він сам усе закодує і відмасштабує)
    prediction = model.predict(input_data)[0]
    # Обмежуємо бал в межах від 0 до 100 на випадок незначних математичних коливань лінійної регресії
    final_score = max(0, min(100, round(prediction, 1)))
    
    # 3. Повертаємо ту саму сторінку, але вже з виведеним результатом
    return get_html_template(result=final_score, form_data=input_data.iloc[0].to_dict())

# Функція, що містить HTML-шаблон (Tailwind CSS для гарного дизайну)
def get_html_template(result=None, form_data=None):
    if form_data is None:
        form_data = {}
        
    # Перевірка для вибору збережених значень у селекторах (щоб форма не скидалася)
    def selected(field, value):
        return "selected" if form_data.get(field) == value else ""

    result_block = ""
    if result is not None:
        result_block = f"""
        <div class="mt-6 p-4 bg-green-50 border-l-4 border-green-500 rounded-r-lg text-center animate-bounce">
            <h3 class="text-lg font-semibold text-green-800">Прогнозований бал з математики:</h3>
            <p class="text-4xl font-black text-green-600 mt-1">{result} / 100</p>
        </div>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="uk">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Система прогнозування оцінок</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-100 min-h-screen flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
        <div class="max-w-2xl w-full space-y-8 bg-white p-8 rounded-xl shadow-lg border border-slate-200">
            <div>
                <h2 class="text-center text-3xl font-extrabold text-slate-900">
                    📊 Прогноз оцінки з математики
                </h2>
                <p class="mt-2 text-center text-sm text-slate-600">
                    Введіть дані студента для розрахунку ймовірного балу на основі ML-моделі
                </p>
            </div>
            
            {result_block}

            <form class="mt-8 space-y-6" action="/predict" method="POST">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <label class="block text-sm font-medium text-slate-700">Стать</label>
                        <select name="gender" class="mt-1 block w-full rounded-md border-slate-300 bg-slate-50 p-2.5 border shadow-sm focus:border-indigo-500 focus:ring-indigo-500">
                            <option value="male" {selected('gender', 'male')}>Чоловіча</option>
                            <option value="female" {selected('gender', 'female')}>Жіноча</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-700">Етнічна група (Race/Ethnicity)</label>
                        <select name="race" class="mt-1 block w-full rounded-md border-slate-300 bg-slate-50 p-2.5 border shadow-sm focus:border-indigo-500 focus:ring-indigo-500">
                            <option value="group A" {selected('race/ethnicity', 'group A')}>Group A</option>
                            <option value="group B" {selected('race/ethnicity', 'group B')}>Group B</option>
                            <option value="group C" {selected('race/ethnicity', 'group C')}>Group C</option>
                            <option value="group D" {selected('race/ethnicity', 'group D')}>Group D</option>
                            <option value="group E" {selected('race/ethnicity', 'group E')}>Group E</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-700">Освіта батьків</label>
                        <select name="education" class="mt-1 block w-full rounded-md border-slate-300 bg-slate-50 p-2.5 border shadow-sm focus:border-indigo-500 focus:ring-indigo-500">
                            <option value="some high school" {selected('parental level of education', 'some high school')}>Some High School</option>
                            <option value="high school" {selected('parental level of education', 'high school')}>High School</option>
                            <option value="some college" {selected('parental level of education', 'some college')}>Some College</option>
                            <option value="associate's degree" {selected('parental level of education', 'associate\'s degree')}>Associate's Degree</option>
                            <option value="bachelor's degree" {selected('parental level of education', 'bachelor\'s degree')}>Bachelor's Degree</option>
                            <option value="master's degree" {selected('parental level of education', 'master\'s degree')}>Master's Degree</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-700">Тип обіду (Lunch)</label>
                        <select name="lunch" class="mt-1 block w-full rounded-md border-slate-300 bg-slate-50 p-2.5 border shadow-sm focus:border-indigo-500 focus:ring-indigo-500">
                            <option value="standard" {selected('lunch', 'standard')}>Стандартний</option>
                            <option value="free/reduced" {selected('lunch', 'free/reduced')}>Безкоштовний / Пільговий</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-700">Підготовчі курси</label>
                        <select name="course" class="mt-1 block w-full rounded-md border-slate-300 bg-slate-50 p-2.5 border shadow-sm focus:border-indigo-500 focus:ring-indigo-500">
                            <option value="none" {selected('test preparation course', 'none')}>Не проходив</option>
                            <option value="completed" {selected('test preparation course', 'completed')}>Завершив успішно</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-700">Бал за читання (0-100)</label>
                        <input type="number" name="reading_score" min="0" max="100" required 
                               value="{form_data.get('reading score', 70)}"
                               class="mt-1 block w-full rounded-md border-slate-300 bg-slate-50 p-2 border shadow-sm focus:border-indigo-500 focus:ring-indigo-500">
                    </div>

                    <div class="md:col-span-2">
                        <label class="block text-sm font-medium text-slate-700">Бал за письмо (0-100)</label>
                        <input type="number" name="writing_score" min="0" max="100" required 
                               value="{form_data.get('writing score', 70)}"
                               class="mt-1 block w-full rounded-md border-slate-300 bg-slate-50 p-2 border shadow-sm focus:border-indigo-500 focus:ring-indigo-500">
                    </div>
                </div>

                <div>
                    <button type="submit" class="group relative w-full flex justify-center py-3 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors">
                        📊 Розрахувати прогноз
                    </button>
                </div>
            </form>
        </div>
    </body>
    </html>
    """
    return html_content