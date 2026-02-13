import {
  Controller,
  Post,
  Get,
  Param,
  UseInterceptors,
  UploadedFiles,
  BadRequestException,
} from '@nestjs/common';
import { FilesInterceptor } from '@nestjs/platform-express';
import { ApiTags, ApiOperation, ApiConsumes, ApiBody } from '@nestjs/swagger';
import { JobsService } from './jobs.service';

const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB

@ApiTags('jobs')
@Controller('api/v1/jobs')
export class JobsController {
  constructor(private readonly jobsService: JobsService) {}

  @Post()
  @UseInterceptors(
    FilesInterceptor('files', 20, { limits: { fileSize: MAX_FILE_SIZE } }),
  )
  @ApiOperation({ summary: 'Create a new analysis job (1–20 images)' })
  @ApiConsumes('multipart/form-data')
  @ApiBody({
    schema: {
      type: 'object',
      properties: {
        files: {
          type: 'array',
          items: { type: 'string', format: 'binary' },
          minItems: 1,
          maxItems: 20,
        },
      },
    },
  })
  async create(
    @UploadedFiles() fileList: Express.Multer.File[],
  ): Promise<{ jobId: string; status: string }> {
    return this.jobsService.create(fileList ?? []);
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get job status and results' })
  async findOne(@Param('id') id: string) {
    return this.jobsService.findOne(id);
  }
}
