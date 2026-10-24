from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from spde_covariance.fragments import FragmentLayout
from spde_covariance.observations import Observation


def _layout(**changes):
    nodes = np.linspace(0, 1, 5)
    values = dict(
        nodes=nodes,
        weights=np.array([0.125, 0.25, 0.25, 0.25, 0.125]),
        intervals=((0.0, 0.75), (0.25, 1.0)),
        row_to_nodes=(np.array([0, 1, 2, 3]), np.array([4, 3, 2, 1])),
        evaluation_matrix=np.column_stack((np.ones(5), nodes)),
        basis_id="linear-H",
    )
    return FragmentLayout(**(values | changes))


def _data(layout):
    return tuple(
        Observation(
            float(index), np.ones(len(row)), layout.evaluation_matrix[row], np.eye(len(row)) * 0.1
        )
        for index, row in enumerate(layout.row_to_nodes)
    )


def test_layout_snapshot_ids_cliques_and_coverage():
    nodes = np.linspace(0, 1, 5)
    rows = [np.arange(4), np.arange(1, 5)]
    layout = _layout(nodes=nodes, row_to_nodes=rows)
    identical = _layout(nodes=nodes.copy(), row_to_nodes=[row.copy() for row in rows])
    assert layout.grid_id == identical.grid_id and layout.layout_id == identical.layout_id
    assert not layout.declared_mask[0, 4] and layout.declared_mask[1, 4]
    np.testing.assert_array_equal(layout.cliques[1], [1, 2, 3, 4])
    nodes[:] = -1
    rows[0][:] = 4
    np.testing.assert_array_equal(layout.nodes, np.linspace(0, 1, 5))
    np.testing.assert_array_equal(layout.row_to_nodes[0], np.arange(4))
    for array in (
        layout.nodes,
        layout.weights,
        layout.evaluation_matrix,
        layout.row_to_nodes[0],
        layout.cliques[0],
        layout.declared_mask,
    ):
        assert not array.flags.writeable
    with pytest.raises(FrozenInstanceError):
        layout.basis_id = "other"


def test_row_order_is_preserved_and_actual_operator_maps_are_checked():
    layout = _layout()
    status = layout.validate_observations(
        _data(layout), basis_id="linear-H", require_operator_validation=True
    )
    assert status["operator_maps_validated"] and status["basis_id_checked"]
    wrong = list(_data(layout))
    item = wrong[1]
    wrong[1] = Observation(item.time, item.y, item.operator[::-1], item.noise_covariance)
    with pytest.raises(ValueError, match="point rows"):
        layout.validate_observations(wrong)
    with pytest.raises(ValueError, match="basis_id"):
        layout.validate_observations(_data(layout), basis_id="another-basis")


def test_generic_reference_layout_requires_explicit_strict_map_validation():
    mapped = _layout()
    generic = _layout(evaluation_matrix=None, basis_id=None)
    result = generic.validate_observations(_data(mapped))
    assert not result["operator_maps_validated"] and not result["basis_id_checked"]
    with pytest.raises(ValueError, match="explicit evaluation_matrix"):
        generic.validate_observations(_data(mapped), require_operator_validation=True)


def test_weights_keep_the_physical_measure_and_ids_describe_layout_separately():
    first = _layout()
    second = _layout(weights=first.weights * 3)
    assert np.sum(second.weights) == 3
    assert first.grid_id != second.grid_id
    reordered = _layout(row_to_nodes=first.row_to_nodes[::-1])
    assert first.grid_id == reordered.grid_id and first.layout_id != reordered.layout_id


@pytest.mark.parametrize(
    "changes",
    [
        {"nodes": [0, 0.5, 0.5, 0.75, 1]},
        {"weights": [0, 0.25, 0.25, 0.25, 0.25]},
        {"weights": [1, 1]},
        {"nodes": [False, True]},
        {"intervals": ((0.2, 0.2),)},
        {"intervals": ((2.0, 3.0),)},
        {"row_to_nodes": ([0, 1, 3], [1, 2, 3, 4])},
        {"row_to_nodes": ([0, 1, 2, 2], [1, 2, 3, 4])},
        {"row_to_nodes": ([0.0, 1.0, 2.0, 3.0], [1, 2, 3, 4])},
        {"evaluation_matrix": np.ones((4, 2))},
        {"basis_id": ""},
    ],
)
def test_invalid_or_sparse_fragment_layouts_are_rejected(changes):
    with pytest.raises(ValueError):
        _layout(**changes)
