"""
The ``mlflow.database`` module provides an API for deploying MLflow models to a SQL DB.
"""
from __future__ import print_function

import logging
import os

from contextlib import contextmanager
from datetime import datetime
import sqlalchemy

from mlflow import onnx
from mlflow import sklearn

from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import (INTERNAL_ERROR,
                                          INVALID_PARAMETER_VALUE)
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import experimental
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.tracking.client import MlflowClient

from .schema.model_table import Base as InitialBase, make_deployed_model

_logger = logging.getLogger(__name__)

SUPPORTED_DEPLOYMENT_FLAVORS = [
    onnx.FLAVOR_NAME,
    sklearn.FLAVOR_NAME
]


@experimental
def create(model_uri, db_uri, user_id, table_name="models"):
    """
    Register an MLflow model with a SQL database.
    :param model_uri: The location, in URI format, of the MLflow model used to build the Azure
                      ML deployment image. For example:
                      - ``models:/<model_name>/<model_version>``
    :param db_uri: The URI of the SQL DB to deploy in the form:
                    <dialect>+<driver>://<username>:<password>@<host>:<port>/<database>
    :param user_id: The principal ID that is deploying the model
    :param table_name: The name of the SQL table to deploy the model to.
    If not specified, will use 'models' table.
    """
    engine = sqlalchemy.create_engine(db_uri)

    table = make_deployed_model(table_name)
    table.__table__.create(bind=engine, checkfirst=True)
    InitialBase.metadata.bind = engine
    SessionMaker = sqlalchemy.orm.sessionmaker(bind=engine)
    ManagedSessionMaker = _get_managed_session_maker(SessionMaker)

    with ManagedSessionMaker() as session:
        deployed_model = collect_model_metadata(model_uri, user_id, table)
        session.add(deployed_model)
        session.flush()


def collect_model_metadata(model_uri, user_id, table):
    _, model_name, model_version = model_uri.split("/")
    client = MlflowClient()
    model_metadata = client.get_model_version(model_name, model_version)
    download_uri = client.get_model_version_download_uri(model_name, model_version)

    # Download the model and the associated metadata files locally
    model_path = _download_artifact_from_uri(download_uri)
    model_config_path = os.path.join(model_path, "MLmodel")
    if not os.path.exists(model_config_path):
        raise MlflowException(
            message=(
                "Failed to find MLmodel configuration within the specified model's"
                " root directory."),
            error_code=INVALID_PARAMETER_VALUE)
    model_config = Model.load(model_config_path)
    flavor = _validate_deployment_flavor(model_config)
    flavor_conf = _get_flavor_configuration(model_path=model_path, flavor_name=flavor)
    model_file = os.path.join(model_path, flavor_conf["data"])
    return table(
        model_name=model_name, model_version=model_version,
        model_framework=flavor,
        model_framework_version=flavor_conf[flavor + "_version"],
        model=open(model_file, "rb").read(),
        model_creation_time=datetime.fromtimestamp(model_metadata.creation_timestamp / 1000),
        model_deployment_time=datetime.now(),
        deployed_by=user_id,
        model_description=model_metadata.description,
        run_id=model_metadata.run_id
    )


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


def _validate_deployment_flavor(model_config):
    """
    Checks that the model flavor is a supported deployment flavor and returns the flavor.
    If not, an exception is thrown.
    :param model_config: An MLflow Model object
    """
    for flavor in model_config.flavors:
        if flavor in SUPPORTED_DEPLOYMENT_FLAVORS:
            return flavor
    raise MlflowException(
        message=(
            "The model flavors: `{flavors}` are not supported for deployment."
            " Please use one of the supported flavors: {supported_flavor_names}".format(
                flavors=model_config.flavors,
                supported_flavor_names=SUPPORTED_DEPLOYMENT_FLAVORS)),
        error_code=INVALID_PARAMETER_VALUE)
