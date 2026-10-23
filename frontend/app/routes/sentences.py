"""GET /api/sentences | DELETE /api/sentences | DELETE /api/sentences/last"""

from app.services.predictor import Predictor


def _get_or_500():
    try:
        return Predictor.get(), None
    except Exception as e:
        return None, (500, {"ok": False, "error": str(e)})


def list_sentences():
    predictor, err = _get_or_500()
    if err:
        return err
    return 200, {"ok": True, "data": {"sentence": predictor.snapshot()["sentence"]}}


def clear():
    predictor, err = _get_or_500()
    if err:
        return err
    predictor.clear_sentence()
    return 200, {"ok": True, "data": predictor.snapshot()}


def backspace():
    predictor, err = _get_or_500()
    if err:
        return err
    predictor.backspace()
    return 200, {"ok": True, "data": predictor.snapshot()}
