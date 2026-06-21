from flask import Flask, request, jsonify, render_template, send_from_directory
from core.membership import MEMBERSHIP_REGISTRY, build_mf
from core.tfs import TFS_REGISTRY
import re
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from core.ptimely import (
    compute_ptimely,
    ptimely_possibility,
    extract_trimf_params_from_fuzzyset,
    extract_trapmf_params_from_fuzzyset,
)
from core.defuzz import centroid, support, core

app = Flask(__name__)

SCENARIOS_DIR = Path(__file__).parent / "saved"
SCENARIOS_DIR.mkdir(exist_ok=True)


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9Ѐ-ӿ\s_\-]", "", name).strip()[:64]




_MF_FREE_NEG_INDICES: dict[str, set[int]] = {
    'sigmf':  {0},
    'dsigmf': {0, 2},
    'psigmf': {0, 2},
}


def validate_payload(payload: dict) -> list:
    """Валідація вхідного JSON для /api/evaluate. Повертає список помилок."""
    errors = []
    operations = payload.get("operations", [])
    if not isinstance(operations, list) or len(operations) == 0:
        errors.append("Немає операцій для оцінювання")
        return errors
    if len(operations) > 20:
        errors.append("Перевищено максимум 20 операцій на одну ТФС")

    for i, op in enumerate(operations):
        params = op.get("params", {})
        if not isinstance(params, dict):
            continue
        for role, spec in params.items():
            if not isinstance(spec, dict):
                continue
            mf_type   = spec.get("mf_type")
            mf_params = spec.get("params", [])
            if not isinstance(mf_params, list):
                errors.append(f"Оп {i+1}, {role}: некоректні параметри ФП")
                continue
            try:
                vals = [float(v) for v in mf_params]
            except (TypeError, ValueError):
                errors.append(f"Оп {i+1}, {role}: параметри не є числами")
                continue

            is_prob = role in ("B1", "K11", "K00")
            is_time = role in ("MT", "DT")
            if is_prob and any(v < 0 or v > 1 for v in vals):
                errors.append(f"Оп {i+1}, {role}: ймовірність має бути в [0, 1]")
            if is_time:
                free = _MF_FREE_NEG_INDICES.get(mf_type, set())
                if any(v < 0 for j, v in enumerate(vals) if j not in free):
                    errors.append(f"Оп {i+1}, {role}: час/дисперсія не може бути від'ємним")

            if mf_type == "trimf" and len(vals) == 3:
                a, b, c = vals
                if not (a <= b <= c):
                    errors.append(f"Оп {i+1}, {role} trimf: має бути a ≤ b ≤ c")
            elif mf_type == "trapmf" and len(vals) == 4:
                a, b, c, d = vals
                if not (a <= b <= c <= d):
                    errors.append(f"Оп {i+1}, {role} trapmf: має бути a ≤ b ≤ c ≤ d")
            elif mf_type == "gaussmf" and len(vals) >= 1:
                if vals[0] <= 0:
                    errors.append(f"Оп {i+1}, {role} gaussmf: σ має бути > 0")

    t_dir = payload.get("directive_time")
    if t_dir is not None:
        try:
            if float(t_dir) < 0:
                errors.append("Директивний час не може бути від'ємним")
        except (TypeError, ValueError):
            errors.append("Директивний час: невалідне число")
    return errors


def find_first_at_threshold(t_grid, p_grid, threshold: float = 0.99):
    """Найменше t де крива p_grid вперше досягає threshold. None якщо ніколи."""
    idxs = np.where(np.array(p_grid) >= threshold)[0]
    return float(t_grid[idxs[0]]) if len(idxs) > 0 else None


def get_aggregated_mt_type(operations: list) -> str | None:
    """Визначає узагальнений тип ФП часу по ВСІХ операціях.
    Повертає 'trimf'/'trapmf' лише якщо ВСІ операції з MT мають цей тип.
    Інакше 'mixed' — для нього автоматично обирається possibility.
    """
    mt_types = set()
    for op in operations:
        if not isinstance(op.get("params"), dict):
            continue
        if "MT" in op["params"]:
            spec = op["params"]["MT"]
            if isinstance(spec, dict) and "mf_type" in spec:
                mt_types.add(spec["mf_type"])
    if len(mt_types) == 0:
        return None
    if len(mt_types) == 1:
        return mt_types.pop()
    return "mixed"


def fuzzyset_to_json(fs):
    return {
        "x": fs.x.tolist(),
        "mu": fs.mu.tolist(),
        "centroid": centroid(fs),
        "support": list(support(fs)),
        "core": list(core(fs)),
    }



def get_user_id() -> str:
    """Отримати UUID користувача з заголовка X-User-Id. 'anonymous' якщо нема."""
    uid = request.headers.get('X-User-Id', '').strip()
    if uid and all(c.isalnum() or c == '-' for c in uid) and len(uid) <= 64:
        return uid
    return 'anonymous'


def scenario_filename(user_id: str, name: str):
    """Структура: saved/<user_id>/<sanitized_name>.json"""
    user_dir = SCENARIOS_DIR / user_id
    user_dir.mkdir(exist_ok=True, parents=True)
    return user_dir / f'{sanitize_name(name)}.json'


def list_user_scenario_files(user_id: str):
    user_dir = SCENARIOS_DIR / user_id
    return list(user_dir.glob('*.json')) if user_dir.exists() else []


@app.route("/")
def title_page():
    return render_template("title.html")


@app.route("/app")
def app_page():
    return render_template("index.html")


@app.route("/help")
def help_page():
    return render_template("help.html")


@app.route("/examples/<path:filename>")
def serve_example(filename):
    return send_from_directory("examples", filename)


@app.route("/api/membership-types")
def list_mfs():
    return jsonify(list(MEMBERSHIP_REGISTRY.keys()))


@app.route("/api/tfs-types")
def list_tfs():
    return jsonify(list(TFS_REGISTRY.keys()))


@app.route("/api/mf-preview", methods=["POST"])
def mf_preview():
    spec = request.get_json()
    try:
        mf = build_mf(spec)
        fs = mf.discretize()
        return jsonify({"x": fs.x.tolist(), "mu": fs.mu.tolist()})
    except (KeyError, ValueError, AssertionError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    payload = request.get_json()
    try:
        # Валідація вхідних даних
        val_errors = validate_payload(payload or {})
        if val_errors:
            return jsonify({"error": "; ".join(val_errors),
                            "errors": val_errors}), 400

        tfs_type        = payload["tfs_type"]
        operations      = payload["operations"]
        T_dir           = payload.get("directive_time")
        ptimely_method  = payload.get("ptimely_method", "auto")

        if tfs_type not in TFS_REGISTRY:
            return jsonify({"error": f"Невідомий тип ТФС: {tfs_type}"}), 400

        result = TFS_REGISTRY[tfs_type](operations)
        response = {
            "P":  fuzzyset_to_json(result["P"]),
            "MT": fuzzyset_to_json(result["MT"]),
            "DT": fuzzyset_to_json(result["DT"]),
        }

        if T_dir is not None:
            resolved_method = ptimely_method
            mt_spec_for_analytical = None

            if ptimely_method in ("auto", "analytical"):
                # Для "auto": аналітичний метод доречний лише тоді, коли вхідні
                # ФП часу — trimf або trapmf. Gaussian та інші типи → possibility.
                # Для явного "analytical" — завжди намагаємось витягти параметри
                # і повертаємо 400 якщо не вдається.
                aggregated_type = get_aggregated_mt_type(operations)

                if ptimely_method == "auto" and aggregated_type not in ("trimf", "trapmf"):
                    # Будь-яке змішування або non-trimf/trapmf → possibility
                    resolved_method = "possibility"
                else:
                    # Витягуємо параметри з РЕЗУЛЬТУЮЧОЇ result["MT"]
                    try:
                        a, b, c = extract_trimf_params_from_fuzzyset(result["MT"])
                        mt_spec_for_analytical = {"mf_type": "trimf",
                                                  "params": [a, b, c]}
                        resolved_method = "analytical"
                    except ValueError:
                        try:
                            a, b, c, d = extract_trapmf_params_from_fuzzyset(
                                result["MT"])
                            mt_spec_for_analytical = {"mf_type": "trapmf",
                                                      "params": [a, b, c, d]}
                            resolved_method = "analytical"
                        except ValueError:
                            if ptimely_method == "analytical":
                                return jsonify({
                                    "error": (
                                        "Аналітичний метод доступний лише для "
                                        "трикутних/трапецієдальних результуючих "
                                        "ФП часу"
                                    )
                                }), 400
                            resolved_method = "possibility"

            # Шкала T_дир: носій результуючої ФП ± PADDING_FACTOR × ширина носія.
            # PADDING_FACTOR=1.0: для носія 4 с шкала буде 12 с — добре видно
            # горизонтальну частину по 0 зліва (стиль MATLAB).
            PADDING_FACTOR = 1.0
            mt_res = result["MT"]
            supp_mask = mt_res.mu > 0.001
            mt_left  = float(mt_res.x[supp_mask].min()) if supp_mask.any() else 0.0
            mt_right = float(mt_res.x[supp_mask].max()) if supp_mask.any() else float(T_dir) * 1.5
            mt_width = max(mt_right - mt_left, 1e-6)

            # Особливий випадок: носій починається біля 0 — без запасу зліва
            if mt_left <= mt_width * 0.05:
                t_dir_min_v = 0.0
            else:
                t_dir_min_v = max(0.0, mt_left - mt_width * PADDING_FACTOR)
            t_dir_max_v = mt_right + mt_width * PADDING_FACTOR

            # Гарантуємо що T_дир користувача потрапляє у видимий діапазон
            t_dir_f = float(T_dir)
            if t_dir_f > t_dir_max_v:
                t_dir_max_v = t_dir_f * 1.1
            if t_dir_f < t_dir_min_v:
                t_dir_min_v = max(0.0, t_dir_f * 0.9)

            curve = compute_ptimely(
                result["MT"],
                t_dir_min=t_dir_min_v,
                t_dir_max=t_dir_max_v,
                method=resolved_method,
                mt_spec=mt_spec_for_analytical,
            )
            # Метрика якості — лише для analytical (порівняння з possibility)
            quality_data = None
            if curve.method == "analytical":
                control_p = ptimely_possibility(result["MT"], curve.t_dir)
                rmse = float(np.sqrt(np.mean((curve.p_sv - control_p) ** 2)))
                quality_pct = max(0.0, min(100.0, (1.0 - 2.0 * rmse) * 100.0))
                if quality_pct >= 95:
                    cat, msg = "good",       "Можна довіряти"
                elif quality_pct >= 85:
                    cat, msg = "acceptable", "Прийнятно, але є розбіжності"
                else:
                    cat, msg = "poor",       "Рекомендуємо перейти на possibility"
                quality_data = {
                    "rmse": rmse,
                    "quality_percent": quality_pct,
                    "category": cat,
                    "message": msg,
                    "control_curve": {"method": "possibility",
                                      "p_sv": control_p.tolist()},
                }

            response["P_sv"] = {
                "t_dir":              curve.t_dir.tolist(),
                "p_sv":               curve.p_sv.tolist(),
                "method":             curve.method,
                "value_at_directive": curve.at(float(T_dir)),
                "t_garant":           find_first_at_threshold(
                                          curve.t_dir, curve.p_sv, 0.99),
                "quality":            quality_data,
            }

        return jsonify(response)
    except NotImplementedError as e:
        return jsonify({"error": f"Не реалізовано: {e}"}), 400
    except (KeyError, ValueError, AssertionError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Внутрішня помилка: {e}"}), 500




@app.route('/api/scenarios', methods=['GET'])
def list_scenarios():
    user_id = get_user_id()
    scenarios = []
    for fp in list_user_scenario_files(user_id):
        try:
            with open(fp, encoding='utf-8') as f:
                data = json.load(f)
            scenarios.append({
                'name': fp.stem,
                'tfs_type': data.get('input', {}).get('tfs_type', '?'),
                'saved_at': data.get('saved_at', '?'),
                'n_operations': len(data.get('input', {}).get('operations', [])),
            })
        except Exception:
            continue
    return jsonify(scenarios)


@app.route('/api/scenarios/clear-all', methods=['POST'])
def clear_all_scenarios():
    user_id = get_user_id()
    user_dir = SCENARIOS_DIR / user_id
    deleted = 0
    if user_dir.exists():
        for fp in user_dir.glob('*.json'):
            fp.unlink()
            deleted += 1
    return jsonify({'ok': True, 'deleted': deleted})


@app.route('/api/scenarios/<name>', methods=['GET'])
def get_scenario(name):
    user_id = get_user_id()
    fp = scenario_filename(user_id, name)
    if not fp.exists():
        return jsonify({'error': 'Сценарій не знайдено'}), 404
    with open(fp, encoding='utf-8') as f:
        return jsonify(json.load(f))


@app.route('/api/scenarios', methods=['POST'])
def save_scenario():
    user_id = get_user_id()
    payload = request.get_json()
    name = sanitize_name(payload.get('name', ''))
    if not name:
        return jsonify({'error': "Ім'я сценарію не може бути порожнім"}), 400
    data = {
        'name': name,
        'saved_at': datetime.now().isoformat(timespec='seconds'),
        'user_id': user_id,
        'input': payload.get('input', {}),
        'result': payload.get('result', {}),
    }
    with open(scenario_filename(user_id, name), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return jsonify({'ok': True, 'name': name})


@app.route('/api/scenarios/<name>', methods=['DELETE'])
def delete_scenario(name):
    user_id = get_user_id()
    fp = scenario_filename(user_id, name)
    if not fp.exists():
        return jsonify({'error': 'Сценарій не знайдено'}), 404
    fp.unlink()
    return jsonify({'ok': True})

if __name__ == "__main__":
    app.run(debug=True, port=5000)