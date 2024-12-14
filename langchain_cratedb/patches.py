def patch_sqlalchemy_dialect() -> None:
    """
    Fixes `AttributeError: 'CrateCompilerSA20' object has no attribute 'visit_on_conflict_do_update'`

    TODO: Upstream to `sqlalchemy-cratedb`.
          https://github.com/crate/sqlalchemy-cratedb/issues/186
    """  # noqa: E501
    from sqlalchemy.dialects.postgresql.base import PGCompiler
    from sqlalchemy_cratedb.compiler import CrateCompiler

    CrateCompiler.visit_on_conflict_do_update = PGCompiler.visit_on_conflict_do_update
    CrateCompiler._on_conflict_target = PGCompiler._on_conflict_target


patch_sqlalchemy_dialect()
