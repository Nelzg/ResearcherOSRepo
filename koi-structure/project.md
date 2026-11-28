---
id: ResearcherOSRepo
title: NLCR (NonLinear Channel Representation)
description: Исследовать, насколько точно представление Y ≈ F(R,D,K)Z с разреженными
  коэффициентами Z восстанавливает MIMO-канал, какие ограничения на множители словаря
  снижают ошибку и как операции NLCR соотносятся с attention.
updated: '2026-11-25T20:52:46Z'
format: koi/1
card_tags: []
card_tag_colors: {}
programs:
- id: nlcr
  title: NLCR
  description: NonLinear Channel Representation
git_repo: true
git_sync_branch: koi-project
---

# direction: root

NLCR (NonLinear Channel Representation)

Исследовать, насколько точно представление Y ≈ F(R,D,K)Z с разреженными коэффициентами Z восстанавливает MIMO-канал, какие ограничения на множители словаря снижают ошибку и как операции NLCR соотносятся с attention.

## method: sparse-fit

<!-- koi:edge {"type": "direction", "label": "Найти условия, снижающие ошибку реконструкции"} -->

Разреженная аппроксимация MIMO-канала

Строить F из комплексных экспонент, нормируя каждый столбец по евклидовой норме. Оценивать Z методом сопряжённых градиентов с диагональным перевзвешиванием. Сравнить ограничения на R, D и K, инициализации и оптимизаторы по ошибке реконструкции. Использовать фурье-словарь как контрольный случай и проверить наблюдение о сходстве спектров R и Y.

<!-- koi:kanban board-sparse-fit -->
| backlog | running | done |
| --- | --- | --- |
| Базовая база <!-- id:c-fe001e98 created:2026-11-25T20:52:14Z updated:2026-11-25T20:52:46Z --> |  |  |

## method: attention-comparison

<!-- koi:edge {"type": "direction", "label": "Установить, какие операции NLCR соответствуют attention"} -->

Сопоставление NLCR с attention

Сопоставить произведение RDK с матрицей оценок, построение F — с вычислением весов, а FZ — с применением весов к значениям. Учитывать, что для фиксированного n нормировка F идёт по t, тогда как в произведении FZ суммирование для фиксированного t идёт по n.

<!-- koi:kanban board-attention-comparison -->
| backlog | running | done |
| --- | --- | --- |
