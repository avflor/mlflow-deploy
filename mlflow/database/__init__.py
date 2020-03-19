"""
The ``mlflow.database`` module provides an API for deploying MLflow models to a SQL DB.
"""
from __future__ import print_function

import logging
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from distutils.version import StrictVersion

import sqlalchemy
from sqlalchemy import MetaData, create_engine

from mlflow import onnx, pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import (INTERNAL_ERROR,
                                          INVALID_PARAMETER_VALUE,
                                          RESOURCE_DOES_NOT_EXIST)
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import experimental
from mlflow.utils.model_utils import _get_flavor_configuration

from .schema.model_table import Base as InitialBase, make_deployed_model

_logger = logging.getLogger(__name__)

SUPPORTED_DEPLOYMENT_FLAVORS = [
    onnx.FLAVOR_NAME
]


@experimental
def create(model_uri, db_uri, flavor, table_name="models"):
    """
    Register an MLflow model with a SQL database.
    :param model_uri: The location, in URI format, of the MLflow model used to build the Azure
                      ML deployment image. For example:
                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``
                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
                      artifact-locations>`_.
    :param db_uri: The URI of the SQL DB to deploy in the form:
                    <dialect>+<driver>://<username>:<password>@<host>:<port>/<database>
    :param flavor: The name of the flavor of the model to use for deployment. Must be ``onnx``,. If the
                   specified flavor is not present or not supported for deployment, an exception
                   will be thrown.
    :param table_name: The name of the SQL table to deploy the model to. If not specified, will use 'models' table.
    """
    engine = sqlalchemy.create_engine(db_uri)

    # insp = sqlalchemy.inspect(engine)
    # expected_tables = set([
    #    table_name
    # ])
    # if len(expected_tables & set(insp.get_table_names())) == 0:
    _logger.info("Creating initial MLflow database tables...")
    table = make_deployed_model(table_name)
    table.__table__.create(bind=engine, checkfirst=True)
    InitialBase.metadata.bind = engine
    SessionMaker = sqlalchemy.orm.sessionmaker(bind=engine)
    ManagedSessionMaker = _get_managed_session_maker(SessionMaker)

    # model_path = _download_artifact_from_uri(model_uri)
    # model_config_path = os.path.join(model_path, "MLmodel")
    # if not os.path.exists(model_config_path):
    #     raise MlflowException(
    #         message=(
    #             "Failed to find MLmodel configuration within the specified model's"
    #             " root directory."),
    #         error_code=INVALID_PARAMETER_VALUE)
    # model_config = Model.load(model_config_path)
    # _validate_deployment_flavor(model_config, flavor)
    # flavor_conf = _get_flavor_configuration(model_path=model_path, flavor_name=flavor)
    # onnx_model = os.path.join(model_path, flavor_conf["data"])

    with ManagedSessionMaker() as session:
        deployed_model = table(
            model_name="name", model_version="1.0",
            model_framework="t",
            model_framework_version="t",
            model=None,
            model_creation_time=None,
            model_deployment_time=None,
            deployed_by=None,
            model_description=None,
            experiment_name=None
        )
        session.add(deployed_model)
        session.flush()


def _initialize_table(engine, table_name):
    _logger.info("Creating initial MLflow database tables...")
    make_deployed_model(table_name).__table__.create(bind=engine,
                                                     checkfirst=True)
    # InitialBase.metadata.create_all(engine)


def _get_managed_session_maker(SessionMaker):
    """
    Creates a factory for producing exception-safe SQLAlchemy sessions that are made available
    using a context manager. Any session produced by this factory is automatically committed
    if no exceptions are encountered within its associated context. If an exception is
    encountered, the session is rolled back. Finally, any session produced by this factory is
    automatically closed when the session's associated context is exited.
    """

    @contextmanager
    def make_managed_session():
        """Provide a transactional scope around a series of operations."""
        session = SessionMaker()
        try:
            yield session
            session.commit()
        except MlflowException:
            session.rollback()
            raise
        except Exception as e:
            session.rollback()
            raise MlflowException(message=e, error_code=INTERNAL_ERROR)
        finally:
            session.close()

    return make_managed_session


def _validate_deployment_flavor(model_config, flavor):
    """
    Checks that the specified flavor is a supported deployment flavor
    and is contained in the specified model. If one of these conditions
    is not met, an exception is thrown.
    :param model_config: An MLflow Model object
    :param flavor: The deployment flavor to validate
    """
    if flavor not in SUPPORTED_DEPLOYMENT_FLAVORS:
        raise MlflowException(
            message=(
                "The specified flavor: `{flavor_name}` is not supported for deployment."
                " Please use one of the supported flavors: {supported_flavor_names}".format(
                    flavor_name=flavor,
                    supported_flavor_names=SUPPORTED_DEPLOYMENT_FLAVORS)),
            error_code=INVALID_PARAMETER_VALUE)
    elif flavor not in model_config.flavors:
        raise MlflowException(
            message=("The specified model does not contain the specified deployment flavor:"
                     " `{flavor_name}`. Please use one of the following deployment flavors"
                     " that the model contains: {model_flavors}".format(
                flavor_name=flavor, model_flavors=model_config.flavors.keys())),
            error_code=RESOURCE_DOES_NOT_EXIST)


def insert_db(db_uri, table_name, column_name, onnx_model):
    engine = sqlalchemy.create_engine(db_uri)
    artifact_content = open(onnx_model, "rb").read()
    # Create connection
    conn = engine.connect()
    meta = MetaData(engine, reflect=True)
    table = meta.tables[table_name]
    # insert data via insert() construct
    ins = table.insert().values(
        model=artifact_content)
    conn.execute(ins)
    # Close connection
    conn.close()
