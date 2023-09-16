import torch
from torch_geometric.typing import Adj, Size, SparseTensor
from torch_geometric.utils import (
    is_sparse)

FUSE_AGGRS = {'add', 'sum', 'mean', 'min', 'max'}

def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
    r"""The initial call to start propagating messages.

    Args:
        edge_index (torch.Tensor or SparseTensor): A :class:`torch.Tensor`,
            a :class:`torch_sparse.SparseTensor` or a
            :class:`torch.sparse.Tensor` that defines the underlying
            graph connectivity/message passing flow.
            :obj:`edge_index` holds the indices of a general (sparse)
            assignment matrix of shape :obj:`[N, M]`.
            If :obj:`edge_index` is a :obj:`torch.Tensor`, its :obj:`dtype`
            should be :obj:`torch.long` and its shape needs to be defined
            as :obj:`[2, num_messages]` where messages from nodes in
            :obj:`edge_index[0]` are sent to nodes in :obj:`edge_index[1]`
            (in case :obj:`flow="source_to_target"`).
            If :obj:`edge_index` is a :class:`torch_sparse.SparseTensor` or
            a :class:`torch.sparse.Tensor`, its sparse indices
            :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
            and :obj:`col = edge_index[0]`.
            The major difference between both formats is that we need to
            input the *transposed* sparse adjacency matrix into
            :meth:`propagate`.
        size ((int, int), optional): The size :obj:`(N, M)` of the
            assignment matrix in case :obj:`edge_index` is a
            :class:`torch.Tensor`.
            If set to :obj:`None`, the size will be automatically inferred
            and assumed to be quadratic.
            This argument is ignored in case :obj:`edge_index` is a
            :class:`torch_sparse.SparseTensor` or
            a :class:`torch.sparse.Tensor`. (default: :obj:`None`)
        **kwargs: Any additional data which is needed to construct and
            aggregate messages, and to update node embeddings.
    """
    decomposed_layers = 1 if self.explain else self.decomposed_layers

    for hook in self._propagate_forward_pre_hooks.values():
        res = hook(self, (edge_index, size, kwargs))
        if res is not None:
            edge_index, size, kwargs = res

    size = self._check_input(edge_index, size)

    # Run "fused" message and aggregation (if applicable).
    if is_sparse(edge_index) and self.fuse and not self.explain:
        coll_dict = self._collect(self._fused_user_args, edge_index, size,
                                  kwargs)

        msg_aggr_kwargs = self.inspector.distribute(
            'message_and_aggregate', coll_dict)
        for hook in self._message_and_aggregate_forward_pre_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs))
            if res is not None:
                edge_index, msg_aggr_kwargs = res
        out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
        for hook in self._message_and_aggregate_forward_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs), out)
            if res is not None:
                out = res

        update_kwargs = self.inspector.distribute('update', coll_dict)
        out = self.update(out, **update_kwargs)

    else:  # Otherwise, run both functions in separation.
        if decomposed_layers > 1:
            user_args = self._user_args
            decomp_args = {a[:-2] for a in user_args if a[-2:] == '_j'}
            decomp_kwargs = {
                a: kwargs[a].chunk(decomposed_layers, -1)
                for a in decomp_args
            }
            decomp_out = []

        for i in range(decomposed_layers):
            if decomposed_layers > 1:
                for arg in decomp_args:
                    kwargs[arg] = decomp_kwargs[arg][i]

            coll_dict = self._collect(self._user_args, edge_index, size,
                                      kwargs)

            msg_kwargs = self.inspector.distribute('message', coll_dict)
            for hook in self._message_forward_pre_hooks.values():
                res = hook(self, (msg_kwargs,))
                if res is not None:
                    msg_kwargs = res[0] if isinstance(res, tuple) else res
            out = self.message(**msg_kwargs)
            for hook in self._message_forward_hooks.values():
                res = hook(self, (msg_kwargs,), out)
                if res is not None:
                    out = res

            if self.explain:
                explain_msg_kwargs = self.inspector.distribute(
                    'explain_message', coll_dict)
                out = self.explain_message(out, **explain_msg_kwargs)

            aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
            for hook in self._aggregate_forward_pre_hooks.values():
                res = hook(self, (aggr_kwargs,))
                if res is not None:
                    aggr_kwargs = res[0] if isinstance(res, tuple) else res

            out = self.aggregate(out, **aggr_kwargs)

            for hook in self._aggregate_forward_hooks.values():
                res = hook(self, (aggr_kwargs,), out)
                if res is not None:
                    out = res

            update_kwargs = self.inspector.distribute('update', coll_dict)
            out = self.update(out, **update_kwargs)

            if decomposed_layers > 1:
                decomp_out.append(out)

        if decomposed_layers > 1:
            out = torch.cat(decomp_out, dim=-1)

    for hook in self._propagate_forward_hooks.values():
        res = hook(self, (edge_index, size, kwargs), out)
        if res is not None:
            out = res

    return out