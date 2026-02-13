import { Injectable, NotFoundException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Label } from './entities/label.entity';

export interface CreateLabelDto {
  filename: string;
  roomType: string;
  amenities: string[];
  features: string[];
  modelPredictions?: object;
}

export interface UpdateLabelDto {
  roomType?: string;
  amenities?: string[];
  features?: string[];
}

@Injectable()
export class LabelsService {
  constructor(
    @InjectRepository(Label)
    private labelsRepository: Repository<Label>,
  ) {}

  async create(dto: CreateLabelDto): Promise<Label> {
    const existing = await this.labelsRepository.findOne({
      where: { filename: dto.filename },
    });
    if (existing) {
      existing.roomType = dto.roomType;
      existing.amenities = dto.amenities;
      existing.features = dto.features;
      existing.modelPredictions = dto.modelPredictions ?? existing.modelPredictions;
      existing.updated_at = new Date();
      return this.labelsRepository.save(existing);
    }
    const label = this.labelsRepository.create({
      ...dto,
      modelPredictions: dto.modelPredictions ?? null,
    });
    return this.labelsRepository.save(label);
  }

  async update(id: number, dto: UpdateLabelDto): Promise<Label> {
    const label = await this.labelsRepository.findOne({ where: { id } });
    if (!label) throw new NotFoundException('Label not found');
    if (dto.roomType !== undefined) label.roomType = dto.roomType;
    if (dto.amenities !== undefined) label.amenities = dto.amenities;
    if (dto.features !== undefined) label.features = dto.features;
    label.updated_at = new Date();
    return this.labelsRepository.save(label);
  }

  async findAll(): Promise<Label[]> {
    return this.labelsRepository.find({ order: { id: 'ASC' } });
  }

  async findByFilename(filename: string): Promise<Label | null> {
    return this.labelsRepository.findOne({ where: { filename } });
  }

  async findOne(id: number): Promise<Label> {
    const label = await this.labelsRepository.findOne({ where: { id } });
    if (!label) throw new NotFoundException('Label not found');
    return label;
  }

  async remove(id: number): Promise<void> {
    await this.labelsRepository.delete(id);
  }

  async exportAll(): Promise<object[]> {
    const labels = await this.labelsRepository.find({ order: { id: 'ASC' } });
    return labels.map((label) => ({
      filename: label.filename,
      roomType: label.roomType,
      amenities: label.amenities,
      features: label.features,
    }));
  }
}
