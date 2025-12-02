/* Seed a small pool of load-test users directly via Prisma. */
'use strict';

const { PrismaClient } = require('@prisma/client');
const bcrypt = require('bcryptjs');

const prisma = new PrismaClient();

async function main() {
  const passwordPlain = process.env.LOADTEST_USER_PASSWORD || 'TestPassword123!';
  const saltRounds = 12;
  const passwordHash = await bcrypt.hash(passwordPlain, saltRounds);

  const users = Array.from({ length: 5 }, (_, idx) => {
    const i = idx + 1;
    return {
      email: `loadtest_user_${i}@loadtest.local`,
      username: `loadtest_user_${i}`,
    };
  });

  for (const user of users) {
    const existing = await prisma.user.findUnique({
      where: { email: user.email },
    });

    if (existing) {
      console.log(`User ${user.email} already exists (id=${existing.id}), skipping`);
      continue;
    }

    const created = await prisma.user.create({
      data: {
        email: user.email,
        username: user.username,
        passwordHash,
        // Other fields use model defaults (role, rating, etc.)
      },
    });
    console.log(`Created user ${created.email} (id=${created.id})`);
  }
}

main()
  .catch((err) => {
    console.error('Error seeding load-test users:', err);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });