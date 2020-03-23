from sqlalchemy import (
    Column, String, VARBINARY, Integer, PrimaryKeyConstraint)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.mssql import DATETIME2

Base = declarative_base()


def make_deployed_model(name):
    class DeployedModel(Base):
        """
        DB model for deployed models.
        """

        __tablename__ = name

        model_id = Column(Integer, autoincrement=True)
        """
        Model ID: `Integer`. *Primary Key* for the table.
        """
        model_name = Column(String(256), nullable=False)
        """
        Model Name as registered in the MLflow registry: ``String` (limit 256 characters).
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
        Framework version used to train the model: : `String`. 
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
        Model deployment time : `DATETIME2`.
        """
        deployed_by = Column(Integer, nullable=True)
        """
        Principal ID that deployed the model: `Integer`. 
        """
        model_description = Column(String(1024), nullable=True)
        """
        Model description as presented in the model registry: `String`.
        """
        run_id = Column(String(100), nullable=True)
        """
        MLflow run id associated with the model: `String`. 
        """

        __table_args__ = (
            PrimaryKeyConstraint('model_id', name='model_pk_' + name),
        )

        def __repr__(self):
            return '<DeployedModel ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})>'.format(
                self.model_id, self.model_name,
                self.model_version, self.model_framework,
                self.model_framework_version, self.model,
                self.model_creation_time,
                self.model_deployment_time,
                self.deployed_by,
                self.model_description,
                self.experiment_name)

    return DeployedModel
