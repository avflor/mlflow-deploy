from sqlalchemy import (
    Column, String, VARBINARY, BigInteger, Integer, PrimaryKeyConstraint)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.mssql import DATETIME2

from mlflow.entities import FileInfo
import os

Base = declarative_base()


class ModelTable(Base):
    """
    DB model for :py:class:`mlflow.entities.ModelTable`. These are recorded in ``models`` table.
    """
    __tablename__ = 'models'
    model_id = Column(Integer, autoincrement=True)
    """
    Model ID: `Integer`. *Primary Key* for ``artifact`` table.
    """
    model_name = Column(String(256), nullable=False)
    """
    Model Name: ``String` (limit 256 characters).
    """
    model_version = Column(String(50), nullable=False)
    """
    Model version: `String`. 
    """
    model_framework = Column(String(50), nullable=False)
    """
    Framework used to train the model: `String` (limit 256 characters).
    """
    model_framework_version = Column(String(50), nullable=False)
    """
    Framework version used to train the model: : `String`. Defined as *Non null* in table schema.
    """
    model = Column(VARBINARY, nullable=False)
    """
    Model  : `VARBINARY`. 
    """
    model_creation_time = Column(DATETIME2, nullable=True)
    """
    Model creation time : `DATETIME2`. Defined as *null* in table schema.
    """
    model_deployment_time = Column(DATETIME2, nullable=False)
    """
    Model deployment time : `DATETIME2`. Defined as *null* in table schema.
    """
    deployed_by = Column(Integer, nullable=True)
    """
    Principal ID that deployed the model: `Integer`. 
    """
    model_description = Column(String(256), nullable=True)
    """
    Model description: `String`.
    """
    experiment_name = Column(String(100), nullable=True)
    """
    Experiment name: `String`. 
    """

    __table_args__ = (
        PrimaryKeyConstraint('model_id', name='model_pk'),
    )

    def __repr__(self):
        return '<Model ({}, {}, {}, {}, {})>'.format(self.artifact_id, self.artifact_name,
                                                           self.group_path,
                                                           self.artifact_content,
                                                           self.artifact_initial_size)