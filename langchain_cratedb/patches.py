import warnings

from sqlalchemy_cratedb.compiler import CrateDDLCompiler


def ddl_compiler_visit_create_index(self, create, **kw) -> str:  # type: ignore[no-untyped-def]
    """
    CrateDB does not support `CREATE INDEX` statements.
    """
    warnings.warn(
        "CrateDB does not support `CREATE INDEX` statements, "
        "they will be omitted when generating DDL statements.",
        stacklevel=2,
    )
    return "SELECT 1"


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
    CrateDDLCompiler.visit_create_index = ddl_compiler_visit_create_index


patch_sqlalchemy_dialect()
